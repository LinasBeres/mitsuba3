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
        SurfaceInteraction3f si;
        Spectrum beta;

        // LightPath(const SurfaceInteraction3f &si, Spectrum s) : si(si), beta(s) {}

        const Point3f &p() const { return si.p; }

        const Normal3f &n() const { return si.n; }

        Mask on_surface() const { return si.shape != nullptr; }

        bool is_light(Scene *scene) const { return dr::any_or<true>(si.emitter(scene) != nullptr); }

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

        // std::cerr << "TeSTING\n";

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;

        std::vector<LightPath> sensor_paths(m_max_depth + 2);
        std::vector<LightPath> emitter_paths(m_max_depth + 1);

        // SurfaceInteraction3f si = scene->ray_intersect(
            // ray, +RayFlags::All, [> coherent = <] true, active);
        // Mask valid_ray = active && si.is_valid();

        int n_sensor = generate_sensor_subpath(scene, sampler, ray, m_max_depth + 2, sensor_paths);
        // int n_emitter = generate_emitter_subpath(scene, sampler, m_max_depth + 1, emitter_paths);

        // Currently do not support connecting sensor straight to camera hence why t = 2 not t = 1
        // for (int t = 2; t <= n_sensor; ++t) {
            // for (int s = 0; s <= n_emitter; ++s) {
                // int depth = t + s - 2;
                // if ((s == 1 && t == 1) || depth < 0 || depth > m_max_depth)
                    // continue;
                // Spectrum L_path = connect_paths(scene, sampler, t, s, sensor_paths, emitter_paths);
//
                // result += L_path;
            // }
        // }

        LightPath p = sensor_paths[0];

        active &= p.si.is_valid();
        Mask valid_ray = active && p.si.is_valid();
        if (dr::none_or<false>(active))
            return { result, valid_ray };

        // std::cerr << "RESULT: " << p.beta << "\n";

        result = p.beta;

        // std::cerr << "RESTURNING..\n";

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }

    Spectrum connect_paths(const Scene *scene,
                           Sampler *sampler,
                           UInt32 t,
                           UInt32 s,
                           const std::vector<LightPath> &sensor_paths,
                           const std::vector<LightPath> &emitter_paths) const {
        return 0.f;
    }

    int generate_sensor_subpath(const Scene *scene,
                                   Sampler *sampler,
                                   const Ray3f &ray,
                                   int max_depth,
                                   std::vector<LightPath> &path) const {
        if (max_depth == 0)
            return 0;
        Spectrum beta(1.f);
        // return 0;
        return random_walk(scene, sampler, ray, beta, max_depth - 1, path, 0, TransportMode::Radiance);
    }

    int generate_emitter_subpath(const Scene *scene,
                                    Sampler *sampler,
                                    int max_depth,
                                    std::vector<LightPath> &path) const {
        return 0;
    }

    int random_walk(const Scene *scene,
                       Sampler *sampler,
                       const Ray3f &ray,
                       Spectrum beta,
                       int max_depth,
                       std::vector<LightPath> &path,
                       int depth,
                       TransportMode transport_mode) const
    {
        // if (max_depth == 0)
            // return 0;

        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        Bool active = true;

        int bounces = 0;

        LightPath &lpath = path[bounces];

        struct LoopState {
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            int depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, throughput, result, eta, depth, \
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                active, sampler)
        } ls = {
            ray,
            throughput,
            result,
            eta,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
            sampler
        };

        while(true) {
            SurfaceInteraction3f si =
                scene->ray_intersect(ls.ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ ls.depth == 0u);
            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result = spec_fma(
                    ls.throughput,
                    ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf,
                    ls.result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            // if (dr::none_or<false>(active_next)) {
                // ls.active = active_next;
                // return; // early exit for scalar mode
            // }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            break;
        }


        SurfaceInteraction3f si = scene->ray_intersect(
            ray, +RayFlags::All, /* coherent = */ true, active);
        valid_ray = valid_ray && si.is_valid();

        lpath.si = si;
        lpath.beta = 0.f;

        // ----------------------- Visible emitters -----------------------

        if (!m_hide_emitters) {
            EmitterPtr emitter_vis = si.emitter(scene, active);
            if (dr::any_or<true>(emitter_vis != nullptr))
                lpath.beta += emitter_vis->eval(si, active);
        }

        active &= si.is_valid();
        if (dr::none_or<false>(active))
            return bounces;

        // ----------------------- Emitter sampling -----------------------

        BSDFContext ctx;
        BSDFPtr bsdf = si.bsdf(ray);
        auto flags = bsdf->flags();
        Mask sample_emitter = active && has_flag(flags, BSDFFlags::Smooth);

        if (dr::any_or<true>(sample_emitter)) {
            for (size_t i = 0; i < 1; ++i) {
                Mask active_e = sample_emitter;
                DirectionSample3f ds;
                Spectrum emitter_val;
                std::tie(ds, emitter_val) = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                active_e &= ds.pdf != 0.f;
                if (dr::none_or<false>(active_e))
                    continue;

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo = si.to_local(ds.d);

                /* Determine BSDF value and probability of having sampled
                   that same direction using BSDF sampling. */
                auto [bsdf_val, bsdf_pdf] = bsdf->eval_pdf(ctx, si, wo, active_e);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                Float mis = dr::select(ds.delta, Float(1.f), mis_weight(
                    ds.pdf * (ScalarFloat) 0.5, bsdf_pdf * (ScalarFloat) 0.5) * 1);
                lpath.beta[active_e] += mis * bsdf_val * emitter_val;
            }
        }

        // ------------------------ BSDF sampling -------------------------

        for (size_t i = 0; i < 1; ++i) {
            auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                               sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            Mask active_b = active && dr::any(unpolarized_spectrum(bsdf_val) != 0.f);

            // Trace the ray in the sampled direction and intersect against the scene
            SurfaceInteraction3f si_bsdf =
                scene->ray_intersect(si.spawn_ray(si.to_world(bs.wo)), active_b);

            // Retain only rays that hit an emitter
            EmitterPtr emitter = si_bsdf.emitter(scene, active_b);
            active_b &= (emitter != nullptr);

            if (dr::any_or<true>(active_b)) {
                Spectrum emitter_val = emitter->eval(si_bsdf, active_b);
                Mask delta = has_flag(bs.sampled_type, BSDFFlags::Delta);

                /* Determine probability of having sampled that same
                   direction using Emitter sampling. */
                DirectionSample3f ds(scene, si_bsdf, si);

                Float emitter_pdf =
                    dr::select(delta, 0.f, scene->pdf_emitter_direction(si, ds, active_b));

                lpath.beta[active_b] +=
                    bsdf_val * emitter_val *
                    mis_weight(bs.pdf * (ScalarFloat) 0.5, emitter_pdf * (ScalarFloat) 0.5) *
                    1;
            }
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
