#include "mitsuba/core/spectrum.h"
#include <cstdlib>
#include <iterator>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class BiDirectionalTracer : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    enum class PathType { Camera, Light, Surface };

    struct LightPath {
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Spectrum beta = 0.f;

        Float pdf_fwd = 0.f;
        Float pdf_rev = 0.f;

        Bool delta = false;
        Bool is_camera = false;
        Bool is_light = false;

        BSDFPtr bsdf = nullptr;

        EmitterPtr emitter = nullptr;

        LightPath() = default;
        LightPath(const SurfaceInteraction3f &si, Spectrum s) : si(si), beta(s) {}

        const Point3f &p() const { return si.p; }

        const Normal3f &n() const { return si.n; }

        Mask on_surface() const { return si.shape != nullptr; }

        bool is_infinite_light() const { return false; }

        bool is_delta_light() const { return false; }

        bool is_connectable() const { return true; }

        Spectrum f(const LightPath &next) const {
            Vector3f wo = dr::normalize(next.p() - p());
            BSDFContext ctx;
            return si.bsdf()->eval(ctx, si, wo);
        }

        Spectrum le() const {
            return 0.f;
        }

        static LightPath create_sensor() {
            LightPath lpath = LightPath();
            lpath.delta = true;
            lpath.pdf_fwd = 1.f;
            lpath.is_camera = true;
            lpath.beta = 1.f;

            return lpath;
        }


        static LightPath create_light(const EmitterPtr &emitter, Spectrum beta) {
            LightPath lpath = LightPath();
            lpath.beta = beta;
            lpath.is_light = true;
            lpath.emitter = emitter;

            return lpath;
        }


    };

    BiDirectionalTracer(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };


        Ray3f ray                     = Ray3f(ray_);
        Spectrum result               = 0.f;

        std::vector<LightPath> sensor_paths(m_max_depth);
        std::vector<LightPath> emitter_paths(m_max_depth + 1);

        int n_sensor = generate_sensor_subpath(scene, sampler, ray, m_max_depth, sensor_paths);
        int n_emitter = generate_emitter_subpath(scene, sampler, m_max_depth+1, emitter_paths);

        // if (n_sensor >= 1 && n_emitter >= 5)
            // result += connect_paths(scene, sampler, sensor_paths, emitter_paths, 1, 5);

        for (int s = 1; s <= n_sensor; ++s) {
            for (int e = 0; e <= n_emitter; ++e) {
                result += connect_paths(scene, sampler, sensor_paths, emitter_paths, s, e);
            }
        }

        result = result / (n_sensor * n_emitter);

        Mask valid_ray = true;

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }


    Spectrum connect_paths(const Scene *scene,
                                Sampler *sampler,
                                const std::vector<LightPath> &sensor_paths,
                                const std::vector<LightPath> &emitter_paths,
                                int s, int e) const {

        Spectrum result = 0.f;

        if (e == 0) {
            // Interpret the sensor subpath as a complete path, i.e. direct emission.
            LightPath prev_lpath = sensor_paths[s - 1]; // get throughput from previous hit
            LightPath lpath = sensor_paths[s]; // current hit bsdf.

            if (dr::any_or<true>(lpath.si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, lpath.si, prev_lpath.si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_lpath.delta))
                    em_pdf = scene->pdf_emitter_direction(prev_lpath.si, ds, !prev_lpath.delta);

                Float mis_bsdf = mis_weight(prev_lpath.pdf_fwd, em_pdf);

                result = prev_lpath.beta * ds.emitter->eval(lpath.si, prev_lpath.pdf_fwd > 0.f) * mis_bsdf;
            }
        }  else if (s == 0) {

        } else if (e == 1) {
            // Sample a point on a emitter and connect it to the sensor subpath.
            LightPath prev_lpath = sensor_paths[s - 1]; // get throughput from previous hit
            LightPath lpath = sensor_paths[s]; // current hit bsdf.

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();

            Mask active_em = has_flag(lpath.bsdf->flags(), BSDFFlags::Smooth);

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    lpath.si, sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                Vector3f wo = lpath.si.to_local(ds.d);
                BSDFContext bsdf_ctx;
                auto [bsdf_val, bsdf_pdf] = lpath.bsdf->eval_pdf(bsdf_ctx, lpath.si, wo);

                bsdf_val = lpath.si.to_world_mueller(bsdf_val, -wo, lpath.si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                result[active_em] = prev_lpath.beta * bsdf_val * em_weight * mis_em;
            }
        } else {
            // First check if connectible...
            Spectrum radiance_throughput  = sensor_paths[s - 1].beta;
            Spectrum importance_throughput = emitter_paths[e - 1].beta;
            LightPath lpath = sensor_paths[s];
            LightPath epath = emitter_paths[e];

            Mask active_em = has_flag(lpath.bsdf->flags(), BSDFFlags::Smooth) && has_flag(epath.bsdf->flags(), BSDFFlags::Smooth);

            Mask occluded = scene->ray_test(lpath.si.spawn_ray_to(epath.si.p), active_em);
            if (dr::none(occluded)) {
                Vector3f d = dr::normalize(epath.si.p - lpath.si.p);
                Vector3f wo = lpath.si.to_local(d);

                BSDFContext bsdf_ctx;
                auto [sensor_bsdf_val, sensor_bsdf_pdf] = lpath.bsdf->eval_pdf(bsdf_ctx, lpath.si, wo);

                Vector3f wi = epath.si.to_local(-d);

                bsdf_ctx = BSDFContext(TransportMode::Importance);
                auto [emitter_bsdf_val, emitter_bsdf_pdf] = epath.bsdf->eval_pdf(bsdf_ctx, epath.si, wi);

                Float mis_bsdf = mis_weight(sensor_bsdf_pdf, emitter_bsdf_pdf);

                result = radiance_throughput * sensor_bsdf_val * importance_throughput * emitter_bsdf_val * mis_bsdf;
            }

        }

        return result;
    }

    int generate_sensor_subpath(const Scene *scene,
                                   Sampler *sampler,
                                   const Ray3f &ray,
                                   int max_depth,
                                   std::vector<LightPath> &path) const {
        if (max_depth == 0)
            return 0;

        path[0] = LightPath::create_sensor();

        return random_walk(scene, sampler, ray, 1.f, max_depth, path, TransportMode::Radiance);
    }

    int generate_emitter_subpath(const Scene *scene,
                                    Sampler *sampler,
                                    int max_depth,
                                    std::vector<LightPath> &path) const {
        if (max_depth == 0)
            return 0;

        // Prepare random samples.
        Float wavelength_sample = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            0.f, wavelength_sample, direction_sample, position_sample);

        path[0] = LightPath::create_light(emitter, ray_weight);

        return random_walk(scene, sampler, ray, 1.f, max_depth, path, TransportMode::Importance);
    }

    int random_walk(const Scene *scene,
                       Sampler *sampler,
                       const Ray3f &ray_,
                       Spectrum,
                       int max_depth,
                       std::vector<LightPath> &path,
                       TransportMode transport_mode) const
    {
        if (max_depth == 0)
            return 0;

        // Initial Variables
        Ray3f ray = ray_;
        int bounces = 0;
        BSDFContext bsdf_ctx(transport_mode);

        while(true)
        {
            LightPath &prev_lpath = path[bounces];

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */  bounces == 0u);

            // Escape if we cannot go any further
            Bool active_next = (bounces + 1 < max_depth) && si.is_valid();
            if (dr::none_or<false>(active_next)) {
                break;
            }

            // ---------------------- BSDF sampling ----------------------
            BSDFPtr bsdf = si.bsdf(ray);

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] = bsdf->sample(bsdf_ctx, si, sample_1, sample_2);

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            // --------------------- Prep Current Path ---------------------
            // Update path values;
            LightPath &lpath = path[++bounces];

            lpath.beta = prev_lpath.beta * bsdf_weight;
            if (transport_mode == TransportMode::Importance) {
                // Using geometric normals (wo points to the camera)
                Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                      wo_dot_geo_n = dr::dot(si.n, si.to_world(bsdf_sample.wo));

                // TODO.
                // Prevent light leaks due to shading normals
                Mask active = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                             (wo_dot_geo_n * Frame3f::cos_theta(bsdf_sample.wo) > 0.f);

                // Adjoint BSDF for shading normals -- [Veach, p. 155]
                Float correction = dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                           (Frame3f::cos_theta(bsdf_sample.wo) * wi_dot_geo_n));
                lpath.beta *= correction;
            }
            lpath.si = si;
            lpath.bsdf = bsdf;
            lpath.pdf_fwd = bsdf_sample.pdf;
            lpath.delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
        }
        return bounces;
    }


    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("BiDirectionalTracer[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(BiDirectionalTracer, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(BiDirectionalTracer, "Bi-Directional Tracer integrator");
NAMESPACE_END(mitsuba)
