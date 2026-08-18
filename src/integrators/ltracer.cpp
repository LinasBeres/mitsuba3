// View-independent light tracer integrator for architectural illuminance evaluation.

#include "mitsuba/core/spectrum.h"
#include <algorithm>
#include <cstdint>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/mesh.h>

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
                     EmitterPtr, BSDF, BSDFPtr, Shape, ShapePtr, Mesh)

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
            DynamicBuffer<Float> adjoint; // δ = ∂L/∂flux per pixel (constant-memory backward pass)

            // Planar-projection fallback for meshes without UV texture coordinates.
            // uv_axis0/1 are the two world axes (0=X,1=Y,2=Z) that map to U and V.
            // uv_origin and uv_inv_size convert world coordinates to [0,1].
            bool         has_uv    = true;
            int          uv_axis0  = 0;
            int          uv_axis1  = 2;
            ScalarFloat  uv_origin0 = 0.f, uv_origin1 = 0.f;
            ScalarFloat  uv_inv_size0 = 1.f, uv_inv_size1 = 1.f;
        };

        std::vector<Lightmap> lightmaps;

        void prepare(const Scene *scene) {
            lightmaps.clear();
            uint32_t first_res = 0;
            for (auto &shape : scene->shapes()) {
                if (!shape->is_receiver())
                    continue;
                uint32_t res = shape->receiver_resolution();
                if (first_res == 0) {
                    first_res = res;
                } else if (res != first_res) {
                    Log(Warn, "ltracer: receiver '%s' requests resolution %u but all "
                              "receivers must share the same resolution for the atlas; "
                              "overriding to %u. Add <integer name=\"receiver_resolution\" "
                              "value=\"%u\"/> to this shape to silence this warning.",
                        shape->id(), res, first_res, first_res);
                    res = first_res;
                }
                Lightmap lm;
                lm.shape        = shape.get();
                lm.width        = res;
                lm.height       = res;
                Float area      = shape->surface_area();
                dr::eval(area);
                lm.surface_area = dr::slice<ScalarFloat>(area, 0u);
                lm.data         = dr::zeros<DynamicBuffer<Float>>(res * res);

                // Only Mesh subclasses carry per-vertex UV coordinates.
                // Built-in shapes (sphere, rectangle, disk, etc.) never have them,
                // so cast first and fall back to planar projection for everything else.
                const Mesh *mesh = dynamic_cast<const Mesh *>(shape.get());
                lm.has_uv = mesh && mesh->has_vertex_texcoords();
                if (!lm.has_uv) {
                    // Identify the surface's dominant normal axis by finding which
                    // world axis has the smallest bounding-box extent — that axis is
                    // perpendicular to the surface.  The other two become U and V.
                    auto bb = shape->bbox();
                    ScalarVector3f ext = bb.max - bb.min;
                    int normal_axis = 0;
                    if (ext[1] < ext[normal_axis]) normal_axis = 1;
                    if (ext[2] < ext[normal_axis]) normal_axis = 2;
                    lm.uv_axis0  = (normal_axis + 1) % 3;
                    lm.uv_axis1  = (normal_axis + 2) % 3;
                    lm.uv_origin0    = bb.min[lm.uv_axis0];
                    lm.uv_origin1    = bb.min[lm.uv_axis1];
                    lm.uv_inv_size0  = (ext[lm.uv_axis0] > 0) ? (1.f / ext[lm.uv_axis0]) : 1.f;
                    lm.uv_inv_size1  = (ext[lm.uv_axis1] > 0) ? (1.f / ext[lm.uv_axis1]) : 1.f;
                    Log(Warn, "ltracer: receiver '%s' has no UV coords — "
                              "falling back to planar projection (axes %d, %d)",
                        shape->id(), lm.uv_axis0, lm.uv_axis1);
                }

                lightmaps.push_back(std::move(lm));
            }

            // scene->shapes() is returned in a per-load (hash/merge) order, so
            // without this sort the atlas slice ↔ shape mapping would change from
            // one mi.load_file() to the next. Sort by shape id() to pin a stable,
            // deterministic atlas order. receiver_shape_ids() walks this same
            // vector, so Python sees the identical order (use it, don't re-filter
            // scene.shapes() on the Python side).
            std::sort(lightmaps.begin(), lightmaps.end(),
                      [](const Lightmap &a, const Lightmap &b) {
                          return a.shape->id() < b.shape->id();
                      });

            Log(Info, "ltracer: prepared %zu receiver lightmap(s)", lightmaps.size());
        }

        // Returns the world-position component along axis 0, 1, or 2.
        static Float world_component(const SurfaceInteraction3f &si, int axis) {
            if (axis == 0) return si.p.x();
            if (axis == 1) return si.p.y();
            return si.p.z();
        }

        // Map a surface interaction to this lightmap's [0,1]^2 UV. For receivers
        // without texcoords we project world position onto the two non-normal
        // axes — note this makes uv a function of si.p, so an si.p attached to the
        // emitter ray yields an attached uv (the hook for position/tilt grads).
        Point2f uv_for(const SurfaceInteraction3f &si, size_t i) const {
            if (lightmaps[i].has_uv)
                return si.uv;
            Float u = (world_component(si, lightmaps[i].uv_axis0)
                       - lightmaps[i].uv_origin0) * lightmaps[i].uv_inv_size0;
            Float v = (world_component(si, lightmaps[i].uv_axis1)
                       - lightmaps[i].uv_origin1) * lightmaps[i].uv_inv_size1;
            return Point2f(u, v);
        }

        // Bilinear splat of scalar lux into the lightmap `data` (forward primal).
        //
        // Deposits to the 4 pixels around the continuous UV, weighted by the
        // fractional position, instead of a single nearest pixel. This makes the
        // lightmap a SMOOTH function of uv — which is exactly what lets the
        // adjoint (gather()) carry a nonzero gradient w.r.t. uv, i.e. w.r.t.
        // emitter position/tilt. A nearest-pixel (integer floor) splat has zero
        // uv-derivative and kills all geometry gradients. Weights sum to 1, so
        // this is energy-conserving except for taps that fall off the map edge
        // (masked out).
        void put(const SurfaceInteraction3f &si, const Spectrum &value, const Mask &active) {
            Float lum = luminance(unpolarized_spectrum(value), si.wavelengths, active);
            for (size_t i = 0; i < lightmaps.size(); ++i) {
                Mask on_this = active && (si.shape == ShapePtr(lightmaps[i].shape));

                int Wp = (int) lightmaps[i].width, Hp = (int) lightmaps[i].height;
                Point2f uv = uv_for(si, i);

                // Continuous pixel coordinate (pixel centers at integer + 0.5).
                Float fx = uv.x() * float(Wp) - 0.5f,
                      fy = uv.y() * float(Hp) - 0.5f;
                Float x0f = dr::floor(fx), y0f = dr::floor(fy);
                Float wx1 = fx - x0f, wy1 = fy - y0f;   // fractional part
                Float wx0 = 1.f - wx1, wy0 = 1.f - wy1;
                Int32 x0 = Int32(x0f), y0 = Int32(y0f);

                for (int t = 0; t < 4; ++t) {
                    Int32 xx = x0 + (t & 1), yy = y0 + (t >> 1);
                    Float w  = ((t & 1) ? wx1 : wx0) * ((t >> 1) ? wy1 : wy0);
                    Mask in_bounds = (xx >= 0) && (xx < Wp) && (yy >= 0) && (yy < Hp);
                    Int32 cx = dr::maximum(dr::minimum(xx, Wp - 1), 0);
                    Int32 cy = dr::maximum(dr::minimum(yy, Hp - 1), 0);
                    UInt32 idx = UInt32(cy * Wp + cx);
                    dr::scatter_add(lightmaps[i].data, w * lum, idx, on_this && in_bounds);
                }
            }
        }

        // Adjoint-transpose of put(): bilinear READ of δ at the continuous UV.
        //
        //   result = Σ_taps w_k(uv) · δ[tap_k]
        //
        // The weights w_k are attached to uv (so ∂result/∂uv ≠ 0 — the spatial
        // redistribution signal), while δ is a detached constant. The tap layout,
        // weights, and bounds MUST match put() exactly, or the gradient is a
        // biased estimate of the wrong quantity. The floor is detached so only
        // the fractional weights (not the integer cell choice) carry gradient.
        Float gather(const SurfaceInteraction3f &si, const Mask &active) const {
            Float result = 0.f;
            for (size_t i = 0; i < lightmaps.size(); ++i) {
                Mask on_this = active && (si.shape == ShapePtr(lightmaps[i].shape));

                int Wp = (int) lightmaps[i].width, Hp = (int) lightmaps[i].height;
                Point2f uv = uv_for(si, i);

                Float fx = uv.x() * float(Wp) - 0.5f,
                      fy = uv.y() * float(Hp) - 0.5f;
                Float x0f = dr::detach(dr::floor(fx)), y0f = dr::detach(dr::floor(fy));
                Float wx1 = fx - x0f, wy1 = fy - y0f;
                Float wx0 = 1.f - wx1, wy0 = 1.f - wy1;
                Int32 x0 = Int32(x0f), y0 = Int32(y0f);

                for (int t = 0; t < 4; ++t) {
                    Int32 xx = x0 + (t & 1), yy = y0 + (t >> 1);
                    Float w  = ((t & 1) ? wx1 : wx0) * ((t >> 1) ? wy1 : wy0);
                    Mask in_bounds = (xx >= 0) && (xx < Wp) && (yy >= 0) && (yy < Hp);
                    Int32 cx = dr::maximum(dr::minimum(xx, Wp - 1), 0);
                    Int32 cy = dr::maximum(dr::minimum(yy, Hp - 1), 0);
                    UInt32 idx = UInt32(cy * Wp + cx);
                    Float d = dr::gather<Float>(lightmaps[i].adjoint, idx, on_this && in_bounds);
                    result += w * d;
                }
            }
            return result;
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
            for (uint32_t i = 0; i < (uint32_t) N; ++i) {
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
        trace_passes(scene, seed, spp);
        return m_receiver.develop();
    }

    // =========================================================
    // trace_passes() — prepare receivers and shoot all particle passes,
    // accumulating primal flux into each lightmap's `data` buffer.
    //
    // Everything the forward render() does EXCEPT develop(). Factored out so the
    // backward pass can reuse the identical primal trace (Step 1) inside a
    // suspend_grad scope, without re-running atlas normalization.
    // =========================================================

    void trace_passes(Scene *scene, UInt32 seed, uint32_t spp) {
        m_receiver.prepare(scene);

        // Resolve sample count.
        // render() arg overrides the XML property if non-zero; property overrides
        // the mode default if non-zero.
        uint32_t requested = (spp > 0) ? spp : m_default_spp;
        if (requested == 0)
            requested = (m_sample_mode == "per_pixel") ? 64u : (1u << 20);

        // 64-bit: in per_pixel mode this is spp x (sum of all lightmap pixels),
        // which overflows 32 bits easily — 2^21 spp over 8 receivers at 64^2 is
        // exactly 2^36, which truncated to 0 and silently rendered a BLACK
        // atlas with no warning. Keep the product in 64 bits and clamp.
        uint64_t total_samples = requested;
        if (m_sample_mode == "per_pixel") {
            uint64_t total_pixels = 0;
            for (auto &lm : m_receiver.lightmaps)
                total_pixels += (uint64_t) lm.width * lm.height;
            total_samples = (uint64_t) requested * total_pixels;
        }
        if (total_samples == 0)
            Throw("ltracer: resolved sample count is zero (requested=%u, mode=%s)",
                  requested, m_sample_mode.c_str());

        Log(Info, "ltracer: shooting %llu rays (%u %s, %s mode) with max depth %u",
            (unsigned long long) total_samples, requested,
            m_sample_mode == "per_pixel" ? "spp" : "total",
            m_sample_mode.c_str(),
            m_max_depth);

        // Independent sampler — no sensor required.
        ref<Sampler> sampler =
            PluginManager::instance()->create_object<Sampler>(Properties("independent"));

        ScalarFloat scale = 1.f / ScalarFloat(total_samples);
        uint64_t samples_done = 0;

        while (samples_done < total_samples) {
            uint32_t wavefront_size = (uint32_t) std::min(
                (uint64_t) m_samples_per_pass, total_samples - samples_done);

            sampler->seed(seed + samples_done, wavefront_size);
            sample(scene, nullptr, sampler.get(), nullptr, scale);

            // Flush pending Dr.JIT kernels and keep memory bounded per pass.
            for (auto &lm : m_receiver.lightmaps)
                dr::eval(lm.data);

            samples_done += wavefront_size;
        }
    }

    // =========================================================
    // render_backward() — reverse-mode derivative of the light trace.
    //
    // Invocation path (verify with the test in main.py):
    //   mi.render(scene, params) installs a _RenderOp dr.CustomOp
    //     → dr.backward(loss) traverses the AD graph
    //     → CustomOp.backward()  (src/python/python/util.py)
    //     → integrator.render_backward(scene, params, grad_in, sensor, seed, spp)
    //     → THIS method.
    //
    // grad_in is dL/d(atlas), same [N, H, W] layout returned by render()/develop().
    //
    // ── Constant-memory adjoint (path-replay structure) ──────────────────────
    // Three stages, none of which retains the full primal path graph, so peak
    // memory is O(atlas size) — independent of spp and path depth:
    //
    //   Step 1  Primal, DETACHED. Trace all particles under suspend_grad,
    //           accumulating primal flux into each lightmap's `data`. No AD
    //           graph is built, so depth/spp cost no adjoint memory.
    //
    //   Step 2  Adjoint of develop(). develop() is a diagonal linear map
    //           (atlas[i,p] = data[i,p] * inv_pixel_area_i), so its adjoint is
    //           the SAME per-pixel rescale of grad_in. We let AD compute it:
    //           enable grad on the (leaf) lightmaps, run develop() once, seed
    //           grad_in on the atlas, traverse back. The result δ_i = grad(data)
    //           is the "adjoint lightmap" — same shape as data, but each pixel
    //           holds ∂L/∂flux instead of luminance. The atlas→surface routing
    //           is exactly the inverse of develop()'s packing scatter.
    //
    //   Pass 2  Attached replay (TODO). Re-trace the SAME particles with emitter
    //           params attached; at each receiver hit GATHER δ_i[pixel] (mirror
    //           of Step 1's scatter_add) and accumulate
    //               proxy += luminance(throughput_attached) * δ_i[pixel]
    //           then dr::backward(proxy) to deposit gradient onto params. Only
    //           the current wavefront's graph is live, so memory stays constant.
    // =========================================================

    void render_backward(Scene *scene, void * /*params*/,
                         const TensorXf &grad_in, Sensor * /*sensor*/,
                         UInt32 seed, uint32_t spp) override {
        Log(Info, "ltracer::render_backward invoked (spp=%u) — constant-memory "
                  "adjoint (one-bounce test path)", spp);

        m_receiver.prepare(scene);

        // Resolve sample count (mirrors trace_passes; the one-bounce test uses a
        // single wavefront so the whole thing replays with one sampler state).
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
        ScalarFloat scale = 1.f / ScalarFloat(total_samples);

        // Set up the sampler HERE and CLONE it before the primal pass. clone()
        // copies the PCG32 state, so the attached replay (Pass 2) re-draws the
        // IDENTICAL emitter samples and reproduces the same first-hit pixels the
        // primal deposited into — which is what makes δ line up with the paths.
        ref<Sampler> sampler =
            PluginManager::instance()->create_object<Sampler>(Properties("independent"));
        sampler->seed(seed, total_samples);
        ref<Sampler> sampler_replay = sampler->clone();

        // ---- Step 1: PRIMAL, detached ------------------------------------------
        // One wavefront, gradients suspended → no path graph retained.
        {
            dr::suspend_grad<Float> no_grad;
            sample(scene, nullptr, sampler.get(), nullptr, scale);
            for (auto &lm : m_receiver.lightmaps)
                dr::eval(lm.data);
        }

        // ---- Step 2: adjoint of develop() → δ ----------------------------------
        // Make each lightmap a differentiable leaf, record only develop(), then
        // back-propagate grad_in through it. δ_i = ∂L/∂data[i] lands per surface.
        for (auto &lm : m_receiver.lightmaps)
            dr::enable_grad(lm.data);

        TensorXf atlas = m_receiver.develop();
        dr::backward_from((atlas * grad_in).array());

        for (auto &lm : m_receiver.lightmaps) {
            lm.adjoint = dr::grad(lm.data);   // the "adjoint lightmap"
            dr::eval(lm.adjoint);
            dr::disable_grad(lm.data);        // detach again; Step 2 graph is done
        }

        // ---- Pass 2: attached replay (one bounce) ------------------------------
        adjoint_pass(scene, sampler_replay.get(), scale);
    }

    // =========================================================
    // adjoint_pass() — one-bounce attached replay.
    //
    // Re-draws the SAME emitter samples as Step 1 (cloned sampler), re-traces to
    // the first hit, gathers δ at that pixel, and back-propagates the deposited
    // luminance to the emitter parameters. No loop: for one bounce all that's
    // needed is prepare_ray + the first intersection.
    //
    // Correctness: the primal deposited  data[pixel] = Σ luminance(w · scale).
    // δ[pixel] = ∂L/∂data[pixel] (a detached constant from Step 2). Each replayed
    // particle contributes  δ[pixel] · luminance(w_attached · scale), and
    //   Σ_particles δ · ∂luminance/∂θ = Σ_pixel δ · ∂data/∂θ = ∂L/∂θ.
    // =========================================================

    void adjoint_pass(const Scene *scene, Sampler *sampler,
                      ScalarFloat sample_scale) const {
        // Emitter parameters (e.g. radiance) must already have grad enabled by
        // the caller. prepare_ray is ATTACHED — the ray weight carries ∂/∂θ.
        auto [ray, throughput] = prepare_ray(scene, sampler);

        Float throughput_max = dr::max(unpolarized_spectrum(throughput));
        Mask active = (throughput_max != 0.f);

        // First (and, for one bounce, only) intersection. si is detached w.r.t.
        // radiance — the pixel it selects does not depend on emitter intensity.
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        active &= si.is_valid() && m_receiver.is_receiver(si.shape);

        // Luminance this particle deposited, attached to the emitter params.
        Spectrum contrib = throughput * sample_scale;
        Float lum = luminance(unpolarized_spectrum(contrib), si.wavelengths, active);

        // δ at the hit pixel — mirror of Pass 1's scatter_add.
        Float delta = m_receiver.gather(si, active);

        // proxy = Σ δ · luminance(contrib); backward deposits ∂L/∂θ onto params.
        Float proxy = dr::select(active, delta * lum, 0.f);
        Float total = dr::sum(proxy);
        dr::backward(total);
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
        Int32 depth = 0;  // depth=0 before first bounce; max_depth=1 means 1 surface hit

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

            // Receiver accumulation.
            // The particle weight w = Le*A*pi already encodes emitter importance.
            // Particle density at a surface point encodes cos(theta_i) implicitly:
            // p(hit dA) = cos(theta_e)/pi * cos(theta_r)/r^2 * dA.
            // Multiplying by cos(theta_r) again gives cos^2 and breaks power
            // conservation. This is equivalent to Jensen's photon mapping formula
            // E_pixel = sum(Phi_k) / pixel_area (no extra cosine).
            Mask on_receiver = m_receiver.is_receiver(si.shape);
            if (dr::any_or<true>(on_receiver)) {
                Spectrum contrib = ls.throughput * sample_scale;
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
