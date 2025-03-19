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

    struct Vertex {
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Spectrum throughput = 0.f;

        Float pdf_fwd = 0.f;
        Float pdf_rev = 0.f;

        Bool delta = false;
        Bool is_camera = false;
        Bool is_light = false;

        BSDFPtr bsdf = nullptr;

        EmitterPtr emitter = nullptr;

        Vertex() = default;
        Vertex(const SurfaceInteraction3f &si, Spectrum s) : si(si), throughput(s) {}

        const Point3f &p() const { return si.p; }

        const Normal3f &n() const { return si.n; }

        Mask on_surface() const { return si.shape != nullptr; }

        bool is_infinite_light() const { return false; }

        bool is_delta_light() const { return false; }

        bool is_connectable() const { return true; }

        Spectrum f(const Vertex &next) const {
            Vector3f wo = dr::normalize(next.p() - p());
            BSDFContext ctx;
            return si.bsdf()->eval(ctx, si, wo);
        }

        Spectrum le() const {
            return 0.f;
        }

        static Vertex create_sensor() {
            Vertex vertex = Vertex();
            vertex.delta = true;
            vertex.pdf_fwd = 1.f;
            vertex.is_camera = true;
            vertex.throughput = 1.f;

            return vertex;
        }


        static Vertex create_light(const EmitterPtr &emitter, Spectrum throughput) {
            Vertex vertex = Vertex();
            vertex.throughput = throughput;
            vertex.is_light = true;
            vertex.emitter = emitter;

            return vertex;
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

        std::vector<Vertex> sensor_paths(m_max_depth);
        std::vector<Vertex> emitter_paths(m_max_depth);

        int n_sensor = generate_sensor_subpath(scene, sampler, ray, m_max_depth, sensor_paths);
        int n_emitter = generate_emitter_subpath(scene, sampler, m_max_depth, emitter_paths);

        // if (n_sensor >= 1 && n_emitter >= 2)
            // result += connect_paths(scene, sampler, sensor_paths, emitter_paths, 1, 2);

        int a = 0;
        for (int s = 0; s <= n_sensor; ++s) {
            for (int e = 0; e <= n_emitter; ++e) {
                int depth = s + e;
                if ((s == 0 && e == 0) || depth <= 0 || depth > m_max_depth)
                    continue;
                Spectrum L = connect_paths(scene, sampler, sensor_paths, emitter_paths, s, e);

                Float result_max = dr::max(unpolarized_spectrum(L));
                if (dr::any(result_max == 0.f)) {
                    continue;
                }

                // TODO: we're at sensor so splat L.
                if (s == 0) {
                } else {
                    result += L;
                    ++a;
                }

            }
        }

        result = result / (a ? a : 1);

        Mask valid_ray = true;

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }


    Spectrum connect_paths(const Scene *scene,
                                Sampler *sampler,
                                const std::vector<Vertex> &sensor_paths,
                                const std::vector<Vertex> &emitter_paths,
                                int s, int e) const {

        Spectrum result = 0.f;

        if (e == 0) {
            // Interpret the sensor subpath as a complete path, i.e. direct emission.
            Vertex prev_vertex = sensor_paths[s - 1]; // get throughput from previous hit
            Vertex vertex = sensor_paths[s]; // current hit bsdf.

            if (dr::any_or<true>(vertex.si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, vertex.si, prev_vertex.si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_vertex.delta))
                    em_pdf = scene->pdf_emitter_direction(prev_vertex.si, ds, !prev_vertex.delta);

                Float mis_bsdf = mis_weight(prev_vertex.pdf_fwd, em_pdf);

                result = vertex.throughput * ds.emitter->eval(vertex.si, prev_vertex.pdf_fwd > 0.f) * mis_bsdf;
            }
        }  else if (s == 0) {

        } else if (e == 1) {
            // Sample a point on a emitter and connect it to the sensor subpath.
            Vertex vertex = sensor_paths[s]; // current hit bsdf.

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();

            Mask active_em = has_flag(vertex.bsdf->flags(), BSDFFlags::Smooth);

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    vertex.si, sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                Vector3f wo = vertex.si.to_local(ds.d);
                BSDFContext bsdf_ctx;
                auto [bsdf_val, bsdf_pdf] = vertex.bsdf->eval_pdf(bsdf_ctx, vertex.si, wo);

                bsdf_val = vertex.si.to_world_mueller(bsdf_val, -wo, vertex.si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                result[active_em] = vertex.throughput * bsdf_val * em_weight * mis_em;
            }
        } else {
            // First check if connectible...
            Vertex s_vertex = sensor_paths[s];
            Vertex e_vertex = emitter_paths[e];

            Mask active = has_flag(s_vertex.bsdf->flags(), BSDFFlags::Smooth) && has_flag(e_vertex.bsdf->flags(), BSDFFlags::Smooth);
            if (dr::none(active))
                return result;

            Mask occluded = scene->ray_test(s_vertex.si.spawn_ray_to(e_vertex.si.p), active);
            if (dr::none(occluded)) {
                // Perp direction.
                Vector3f d     = e_vertex.si.p - s_vertex.si.p;
                Float dist     = dr::norm(d);
                Float inv_dist = dr::rcp(dist);
                d             *= inv_dist;

                // Eval sensor vertex bsdf
                Vector3f wo = s_vertex.si.to_local(d);
                BSDFContext bsdf_ctx;
                auto [sensor_bsdf_val, sensor_bsdf_pdf] = s_vertex.bsdf->eval_pdf(bsdf_ctx, s_vertex.si, wo);

                // Eval emitter vertex bsdf
                Vector3f wi = e_vertex.si.to_local(-d);
                bsdf_ctx = BSDFContext(TransportMode::Importance);
                auto [emitter_bsdf_val, emitter_bsdf_pdf] = e_vertex.bsdf->eval_pdf(bsdf_ctx, e_vertex.si, wi);

                Float mis_bsdf = mis_weight(sensor_bsdf_pdf, emitter_bsdf_pdf);

                result = s_vertex.throughput * sensor_bsdf_val *
                         e_vertex.throughput * emitter_bsdf_val *
                         mis_bsdf * dr::square(inv_dist);
            }

        }

        return result;
    }

    int generate_sensor_subpath(const Scene *scene,
                                   Sampler *sampler,
                                   const Ray3f &ray,
                                   int max_depth,
                                   std::vector<Vertex> &path) const {
        if (max_depth == 0)
            return 0;

        path[0] = Vertex::create_sensor();

        return random_walk(scene, sampler, ray, /* throughput: */ 1.f, max_depth, path, TransportMode::Radiance);
    }

    int generate_emitter_subpath(const Scene *scene,
                                    Sampler *sampler,
                                    int max_depth,
                                    std::vector<Vertex> &path) const {
        if (max_depth == 0)
            return 0;

        // Prepare random samples.
        Float wavelength_sample = sampler->next_1d();
        Point2f direction_sample = sampler->next_2d(),
                position_sample  = sampler->next_2d();

        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            0.f, wavelength_sample, direction_sample, position_sample);

        path[0] = Vertex::create_light(emitter, ray_weight);

        return random_walk(scene, sampler, ray, ray_weight, max_depth, path, TransportMode::Importance);
    }

    int random_walk(const Scene *scene,
                       Sampler *sampler,
                       const Ray3f &ray_,
                       Spectrum throughput,
                       int max_depth,
                       std::vector<Vertex> &path,
                       TransportMode transport_mode) const
    {
        if (max_depth == 0)
            return 0;

        // Initial Variables
        Mask active = true;
        Ray3f ray = ray_;
        int bounces = 0;
        BSDFContext bsdf_ctx(transport_mode);

        while(true)
        {
            // -------------------- Stopping criterion ---------------------
            Float throughput_max = dr::max(unpolarized_spectrum(throughput));
            active &= (throughput_max != 0.f);

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */  bounces == 0u);

            // Escape if we cannot go any further
            Bool active_next = active && (bounces + 1 < max_depth) && si.is_valid();
            if (dr::none_or<false>(active_next)) {
                break;
            }

            // ---------------------- BSDF sampling ----------------------
            BSDFPtr bsdf = si.bsdf(ray);

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] = bsdf->sample(bsdf_ctx, si, sample_1, sample_2);

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            // --------------------- Particle Tracing ---------------------
            Float correction = 1.f;
            if (transport_mode == TransportMode::Importance) {
                // Using geometric normals (wo points to the camera)
                Float wi_dot_geo_n = dr::dot(si.n, -ray.d),
                      wo_dot_geo_n = dr::dot(si.n, si.to_world(bsdf_sample.wo));

                // Prevent light leaks due to shading normals
                active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                          (wo_dot_geo_n * Frame3f::cos_theta(bsdf_sample.wo) > 0.f);

                // Adjoint BSDF for shading normals -- [Veach, p. 155]
                correction = dr::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                           (Frame3f::cos_theta(bsdf_sample.wo) * wi_dot_geo_n));
            }
            if (dr::none_or<false>(active))
                break;

            // ------------------- Update Current Hit --------------------
            Vertex &vertex= path[++bounces];
            vertex.throughput = throughput;
            vertex.si = si;
            vertex.bsdf = bsdf;
            vertex.pdf_fwd = bsdf_sample.pdf;
            vertex.delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // ----------------------- Prepare Ray -----------------------
            throughput *= bsdf_weight * correction;
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
