#include "mitsuba/core/spectrum.h"
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
        Spectrum beta = 0.f;
        Spectrum L = 0.f;

        Float pdf_fwd = 0.f;
        Float pdf_rev = 0.f;

        Bool delta;

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


        static LightPath create_light(const SurfaceInteraction3f& si, Spectrum beta, float pdf) {
            LightPath lpath;
            lpath.si = si;
            lpath.beta = beta;
            lpath.pdf_fwd = pdf;
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

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum result               = 0.f;

        std::vector<LightPath> sensor_paths(m_max_depth);
        std::vector<LightPath> emitter_paths(m_max_depth + 1);

        // SurfaceInteraction3f si = scene->ray_intersect(
            // ray, +RayFlags::All, [> coherent = <] true, active);
        // Mask valid_ray = active && si.is_valid();

        int n_sensor = generate_sensor_subpath(scene, sampler, ray, m_max_depth, sensor_paths);

        LightPath p = sensor_paths[n_sensor];

        Mask valid_ray = active && p.si.is_valid();
        if (dr::none_or<false>(active))
            return { result, valid_ray };

        result = p.L;

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }

    int generate_sensor_subpath(const Scene *scene,
                                   Sampler *sampler,
                                   const Ray3f &ray,
                                   int max_depth,
                                   std::vector<LightPath> &path) const {
        if (max_depth == 0)
            return 0;

        LightPath &lpath = path[0];
        Spectrum beta(1.f);
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        lpath.si = si;
        lpath.beta = beta;
        lpath.L = 0.f;
        lpath.pdf_fwd = 1.f;
        lpath.delta = true;

        return random_walk(scene, sampler, ray, beta, max_depth, path, TransportMode::Radiance);
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
        Bool active = true;
        int bounces = 0;
        BSDFContext bsdf_ctx(transport_mode);

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        LightPath &prev_lpath = path[bounces];

        while(true) {
            // if (dr::any(dr::max(unpolarized_spectrum(beta)) != 0.f))
                // break;
            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */  bounces == 0u);

            // ---------------------- Direct emission ----------------------
            // Did we hit a light?
            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, prev_lpath.si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_lpath.delta))
                    em_pdf = scene->pdf_emitter_direction(prev_lpath.si, ds, !prev_lpath.delta);

                Float mis_bsdf = mis_weight(prev_lpath.pdf_fwd, em_pdf);

                prev_lpath.L = spec_fma(prev_lpath.beta, ds.emitter->eval(si, prev_lpath.pdf_fwd > 0.f) * mis_bsdf, prev_lpath.L);
            }

            Bool active_next = (bounces + 1 < max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                break; // early exit for scalar mode
            }


            BSDFPtr bsdf = si.bsdf(ray);

            // ---------------------- Emitter sampling ----------------------
            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------
            LightPath &lpath = path[bounces + 1];
            lpath.L = 0.f;
            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                lpath.L[active_em] = spec_fma(
                    prev_lpath.beta, bsdf_val * em_weight * mis_em, prev_lpath.L);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            // ------ Update loop variables based on current interaction ------
            lpath.si = si;
            lpath.beta = prev_lpath.beta * bsdf_weight;
            lpath.pdf_fwd = bsdf_sample.pdf;
            lpath.delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            prev_lpath = lpath;
            if (dr::any(si.is_valid()))
                bounces += 1;
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
