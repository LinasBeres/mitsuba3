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

    struct LightPath {
        SurfaceInteraction3f s;
        Spectrum beta;
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
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        std::vector<LightPath> sensor_paths(m_max_depth + 2);
        std::vector<LightPath> emitter_paths(m_max_depth + 1);

        UInt32 n_sensor = generate_sensor_subpath(scene, sampler, ray, m_max_depth + 2, sensor_paths);
        UInt32 n_emitter = generate_emitter_subpath(scene, sampler, m_max_depth + 1, emitter_paths);

        // Currently do not support connecting sensor straight to camera hence why t = 2 not t = 1
        for (UInt32 t = 2; dr::any_or<true>(t <= n_sensor); ++t) {
            for (UInt32 s = 0; dr::any_or<true>(s <= n_emitter); ++s) {
                UInt32 depth = t + s - 2;
                if (dr::any_or<true>((s == 1 && t == 1) || depth < 0 || depth > m_max_depth))
                    continue;
                Spectrum L_path = connect_paths(scene, sampler, t, s, sensor_paths, emitter_paths);

                result += L_path;
            }
        }

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

    UInt32 generate_sensor_subpath(const Scene *scene,
                                   Sampler *sampler,
                                   const RayDifferential3f &ray,
                                   UInt32 max_depth,
                                   std::vector<LightPath> &path) const {
        
        return 0;
    }

    UInt32 generate_emitter_subpath(const Scene *scene,
                                    Sampler *sampler,
                                    UInt32 max_depth,
                                    std::vector<LightPath> &path) const {
        return 0;
    }

    // UInt32 random_walk(const Scene *scene,
                       // Sampler *sampler,
//


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
