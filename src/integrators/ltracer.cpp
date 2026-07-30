// View-independent light tracer integrator for architectural illuminance evaluation.

#include "mitsuba/core/spectrum.h"
#include <cstdint>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * View-independent light tracer.
 *
 * Traces particles from emitters. At each surface bounce, if the shape is
 * tagged as a receiver (is_receiver=true in scene XML), accumulates
 * throughput * cos(theta_i) into a per-shape UV lightmap.
 *
 * No camera. No ImageBlock. No sensor required.
 */
template <typename Float, typename Spectrum>
class LightTracerIntegrator final : public AdjointIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(AdjointIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_max_depth)
    MI_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr, Shape, ShapePtr)

    // =========================================================
    // SurfaceReceiver
    //
    // One UV lightmap per shape tagged <boolean name="is_receiver" value="true"/>.
    // Optional: <integer name="receiver_resolution" value="256"/>
    // =========================================================
    struct SurfaceReceiver {
        struct Lightmap {
            const Shape *shape;
            uint32_t width, height;
            ScalarFloat surface_area;    // world-space area of the shape, for lux normalization
            DynamicBuffer<Float> data;   // accumulated flux per pixel (before normalization)
        };

        std::vector<Lightmap> lightmaps;

        void prepare(const Scene *scene) {
            lightmaps.clear();
            for (auto &shape : scene->shapes()) {
                if (!shape->is_receiver())
                    continue;
                uint32_t res = shape->receiver_resolution();
                Lightmap lm;
                lm.shape        = shape.get();
                lm.width        = res;
                lm.height       = res;
                Float area      = shape->surface_area();
                dr::eval(area);
                lm.surface_area = dr::slice<ScalarFloat>(area, 0u);
                lm.data         = dr::zeros<DynamicBuffer<Float>>(res * res);
                lightmaps.push_back(std::move(lm));
            }
            std::cerr << "Created a light map with length: " << lightmaps.size() << "\n";
        }

        // Accumulates incoming irradiance as scalar lux (pre-BSDF).
        // luminance() handles all variants: Rec.709 weights for RGB,
        // CIE 1931 Y integral for spectral. Target scenes are assumed diffuse.
        void put(const SurfaceInteraction3f &si, const Spectrum &value, const Mask &active) {
            Float lum = luminance(unpolarized_spectrum(value), si.wavelengths, active);
            for (size_t i = 0; i < lightmaps.size(); i++) {
                Mask on_this = active && (si.shape == ShapePtr(lightmaps[i].shape));
                // No any_or guard: inside dr::while_loop recording, any_or<false>
                // evaluates the symbolic mask as false and silently drops the
                // scatter_add from the compiled kernel. scatter_add with a
                // fully-false mask is a no-op at runtime.
                UInt32 px  = dr::minimum(UInt32(si.uv.x() * lightmaps[i].width),  lightmaps[i].width  - 1);
                UInt32 py  = dr::minimum(UInt32(si.uv.y() * lightmaps[i].height), lightmaps[i].height - 1);
                UInt32 idx = py * lightmaps[i].width + px;
                dr::scatter_add(lightmaps[i].data, lum, idx, on_this);
            }
        }

        Mask is_receiver(ShapePtr shape) const {
            Mask result = false;
            for (auto &lm : lightmaps)
                result |= (shape == ShapePtr(lm.shape));
            return result;
        }

        TensorXf develop() {
            if (lightmaps.empty())
                return TensorXf();

            size_t   N      = lightmaps.size();
            uint32_t H      = lightmaps[0].height;  // assumes uniform resolution
            uint32_t W      = lightmaps[0].width;
            uint32_t pixels = H * W;

            // Build interleaved atlas: layout [N, H, W, 3]
            // atlas[ i*pixels*3 + j*3 + c ] = lightmap[i].channel[c][j]
            // Concatenate per-surface lux maps into [N, H, W]
            DynamicBuffer<Float> atlas =
                dr::empty<DynamicBuffer<Float>>(N * pixels);

            // Normalize accumulated flux by pixel area to get irradiance (lux / W/m²).
            // Each pixel covers world-space area = surface_area / (W * H).
            // Irradiance = flux / pixel_area = flux * W*H / surface_area.
            UInt32 pixel_idx = dr::arange<UInt32>(pixels);
            for (uint32_t i = 0; i < (uint32_t) N; i++) {
                ScalarFloat inv_pixel_area =
                    ScalarFloat(lightmaps[i].width * lightmaps[i].height) / lightmaps[i].surface_area;
                UInt32 base = i * pixels;
                dr::scatter(atlas, lightmaps[i].data * inv_pixel_area, pixel_idx + base);
            }

            size_t shape[3] = { N, (size_t) H, (size_t) W };
            return TensorXf(std::move(atlas), 3, shape);
        }
    };

    mutable SurfaceReceiver m_receiver;

    // "total"    — spp arg means total rays fired regardless of scene geometry
    // "per_pixel" — spp arg means rays per receiver pixel; total scales with scene
    std::string m_sample_mode;
    uint32_t    m_default_spp;

    LightTracerIntegrator(const Properties &props) : Base(props) {
        m_sample_mode = props.get<std::string>("sample_mode", "total");
        m_default_spp = (uint32_t) props.get<int>("spp", 0);
    }

    /// Shape IDs of receiver surfaces in atlas order (index i → atlas slice [i, :, :, :])
    std::vector<std::string> receiver_shape_ids() const {
        std::vector<std::string> ids;
        ids.reserve(m_receiver.lightmaps.size());
        for (auto &lm : m_receiver.lightmaps)
            ids.push_back(lm.shape->id());
        return ids;
    }

    // =========================================================
    // render() — sensor parameter is ignored entirely
    // =========================================================

    TensorXf render(Scene *scene, uint32_t sensor_idx, UInt32 seed,
                    uint32_t spp, bool develop, bool evaluate) {
        Sensor *sensor = scene->sensors().empty()
            ? nullptr : scene->sensors()[sensor_idx].get();
        return render(scene, sensor, seed, spp, develop, evaluate);
    }

    TensorXf render(Scene *scene, Sensor * /*ignored*/, UInt32 seed,
                    uint32_t spp, bool /* develop */, bool /* evaluate */) override {
        m_receiver.prepare(scene);

        // Resolve sample count.
        // render() arg overrides the XML property if non-zero; property overrides
        // the mode default if non-zero.
        uint32_t requested = (spp > 0) ? spp : m_default_spp;
        if (requested == 0)
            requested = (m_sample_mode == "per_pixel") ? 64u : (1u << 20);

        uint32_t total_samples = requested;
        if (m_sample_mode == "per_pixel") {
            uint32_t total_pixels = 0;
            for (auto &lm : m_receiver.lightmaps)
                total_pixels += lm.width * lm.height;
            total_samples = requested * total_pixels;
        }

        Log(Info, "ltracer: shooting %u rays (%u %s, %s mode)",
            total_samples, requested,
            m_sample_mode == "per_pixel" ? "spp" : "total",
            m_sample_mode.c_str());
        std::cerr << "ltracer: shooting " << total_samples << " rays ( " << requested << " " << (m_sample_mode == "per_pixel" ? "spp" : "total") << " " << m_sample_mode.c_str() << " mode)\n";

        // Independent sampler — no sensor required.
        ref<Sampler> sampler =
            PluginManager::instance()->create_object<Sampler>(Properties("independent"));

        ScalarFloat scale = 1.f / ScalarFloat(total_samples);
        uint32_t samples_done = 0;

        while (samples_done < total_samples) {
            uint32_t wavefront_size =
                std::min((uint32_t) m_samples_per_pass, total_samples - samples_done);

            sampler->seed(seed + samples_done, wavefront_size);
            sample(scene, nullptr, sampler.get(), nullptr, scale);

            // Flush pending Dr.JIT kernels and keep memory bounded per pass.
            for (auto &lm : m_receiver.lightmaps)
                dr::eval(lm.data);

            samples_done += wavefront_size;
        }

        return m_receiver.develop();
    }

    // =========================================================
    // sample() — one particle from a random emitter
    // Sensor* and ImageBlock* are intentionally unused.
    // =========================================================

    void sample(const Scene *scene, const Sensor * /*ignored*/,
                Sampler *sampler, ImageBlock * /*ignored*/,
                ScalarFloat sample_scale) const override {
        auto [ray, throughput] = prepare_ray(scene, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = (throughput_max != 0.f);

        trace_illuminance(ray, scene, sampler, throughput, sample_scale, active);
    }

    // =========================================================
    // prepare_ray() — sample one ray from a random emitter
    // Same as ptracer but without sensor shutter time.
    // =========================================================

    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           Sampler *sampler) const {
        Float time = 0.f;

        Float wavelength_sample  = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample);

        return { ray, ray_weight };
    }

    // =========================================================
    // trace_illuminance() — main particle bounce loop
    //
    // At each surface hit:
    //   1. If receiver: accumulate throughput * cos(theta_i)
    //   2. BSDF sample next direction
    //   3. Russian roulette
    //
    // connect_sensor() is gone — no shadow ray, no splatting.
    // =========================================================

    void trace_illuminance(Ray3f ray, const Scene *scene, Sampler *sampler,
                           Spectrum throughput, ScalarFloat sample_scale,
                           Mask active) const {
        Float eta(1.f);
        Int32 depth = 1;

        PreliminaryIntersection3f pi =
            scene->ray_intersect_preliminary(ray, active);
        active &= pi.is_valid();
        if (m_max_depth >= 0)
            active &= depth < m_max_depth;

        struct LoopState {
            Bool active;
            Int32 depth;
            Ray3f ray;
            Spectrum throughput;
            PreliminaryIntersection3f pi;
            Float eta;
            Sampler *sampler;

            DRJIT_STRUCT(LoopState, active, depth, ray, throughput, pi, eta, sampler)
        } ls = { active, depth, ray, throughput, pi, eta, sampler };

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState &ls) { return ls.active; },
            [this, scene, sample_scale](LoopState &ls) {

            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

            BSDFPtr bsdf = si.bsdf(ls.ray);

            // Receiver accumulation
            Mask on_receiver = m_receiver.is_receiver(si.shape);
            if (dr::any_or<true>(on_receiver)) {
                Float cos_theta_i = Frame3f::cos_theta(si.wi);
                Spectrum contrib = ls.throughput * dr::maximum(0.f, cos_theta_i) * sample_scale;
                m_receiver.put(si, contrib, ls.active && on_receiver);
            }

            // BSDF sample (identical to ptracer)
            BSDFContext ctx(TransportMode::Importance);
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si,
                             ls.sampler->next_1d(ls.active),
                             ls.sampler->next_2d(ls.active), ls.active);

            // Geometric normal correction — Veach p.155
            Float wi_dot_geo_n = dr::dot(si.n, -ls.ray.d),
                  wo_dot_geo_n = dr::dot(si.n, si.to_world(bs.wo));

            ls.active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                         (wo_dot_geo_n * Frame3f::cos_theta(bs.wo) > 0.f);

            Float correction = dr::abs(
                (Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                (Frame3f::cos_theta(bs.wo) * wi_dot_geo_n));
            ls.throughput *= bsdf_val * correction;
            ls.eta *= bs.eta;

            ls.active &= dr::any(unpolarized_spectrum(ls.throughput) != 0.f);
            if (dr::none_or<false>(ls.active))
                return;

            ls.depth++;
            if (m_max_depth >= 0)
                ls.active &= ls.depth < m_max_depth;

            // Russian roulette (identical to ptracer)
            Mask use_rr = ls.depth > m_rr_depth;
            if (dr::any_or<true>(use_rr)) {
                Float q = dr::minimum(
                    dr::max(unpolarized_spectrum(ls.throughput)) *
                        dr::square(ls.eta),
                    0.95f);
                dr::masked(ls.active, use_rr) &=
                    ls.sampler->next_1d(ls.active) < q;
                dr::masked(ls.throughput, use_rr) *= dr::rcp(q);
            }

            ls.ray = si.spawn_ray(si.to_world(bs.wo));
            ls.pi  = scene->ray_intersect_preliminary(
                ls.ray, false, jit_flag(JitFlag::LoopRecord), 0, 0, ls.active);
            ls.active &= ls.pi.is_valid();
        },
        "Light Tracer Integrator");
    }

    std::string to_string() const override {
        return tfm::format("LightTracerIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MI_DECLARE_CLASS(LightTracerIntegrator)
};

MI_EXPORT_PLUGIN(LightTracerIntegrator)
NAMESPACE_END(mitsuba)
