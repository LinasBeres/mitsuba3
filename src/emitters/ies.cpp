#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>

#include <fstream>
#include <sstream>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-ies:

IES photometric light source (:monosp:`ies`)
--------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Path to an IESNA LM-63 photometric data file (goniometric intensity table).

 * - scale
   - |float|
   - Multiplier on the tabulated candela values (dimming / efficacy factor,
     e.g. Sydney's per-fixture scale_factor). (Default: 1)
   - |exposed|, |differentiable|

 * - to_world
   - |transform|
   - Emitter-to-world transform. In local space the luminaire sits at the origin
     with its photometric NADIR (vertical angle 0) pointing along local -Z.
   - |exposed|

A delta (point) light whose angular radiant intensity follows a tabulated IES
goniometric distribution I(theta, phi) [candela]. Because candela is already a
photometric intensity (lm/sr), illuminance on a receiver is E = I(w)/r^2 in lux —
so this drops straight into the view-independent lux pipeline.

Directions are importance-sampled proportional to I(theta,phi)*sin(theta) (flux
measure), the standard goniometric/envmap scheme.

Phase 1: forward only. `to_world` is non-differentiable (aim derivatives come in
Phase 2 via an attached aim rotation). Type C photometry; rotationally symmetric
(one horizontal angle) or a full explicit horizontal range.
 */

template <typename Float, typename Spectrum>
class IESLight final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_medium, m_to_world)
    MI_IMPORT_TYPES(Scene, Texture)

    using Warp = Marginal2D<Float, 0>;
    using FloatStorage = DynamicBuffer<Float>;

    IESLight(const Properties &props) : Base(props) {
        m_flags  = +EmitterFlags::DeltaPosition;
        m_scale  = props.get<ScalarFloat>("scale", 1.f);

        fs::path file_path = Thread::thread()->file_resolver()->resolve(
            props.get<std::string>("filename"));
        parse_ies(file_path);

        dr::make_opaque(m_scale);
    }

    // ── LM-63 parser ────────────────────────────────────────────────────────
    // Robust to arbitrary line-wrapping: after the TILT line every remaining
    // token is read into one flat number stream and consumed positionally.
    void parse_ies(const fs::path &path) {
        std::ifstream f(path.string());
        if (!f.good())
            Throw("IESLight: could not open \"%s\"", path.string());

        std::string line, tilt;
        bool found_tilt = false;
        while (std::getline(f, line)) {
            // normalize CRLF
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("TILT=", 0) == 0) {
                tilt = line.substr(5);
                found_tilt = true;
                break;
            }
        }
        if (!found_tilt)
            Throw("IESLight: no TILT= line in \"%s\"", path.string());
        if (string::to_lower(string::trim(tilt)) != "none")
            Throw("IESLight: only TILT=NONE is supported (got '%s')", tilt);

        // Slurp all remaining whitespace-separated numbers.
        std::vector<double> n;
        double v;
        while (f >> v) n.push_back(v);

        auto need = [&](size_t k) {
            if (n.size() < k)
                Throw("IESLight: truncated photometric data in \"%s\"", path.string());
        };
        need(13);
        size_t i          = 0;
        /* n_lamps      */   i++;
        /* lumens/lamp  */   i++;
        double cand_mult  = n[i++];
        int    n_vert     = (int) n[i++];
        int    n_horiz    = (int) n[i++];
        /* photom_type  */   i++;   // 1 = Type C (assumed)
        /* units_type   */   i++;
        /* width        */   i++;
        /* length       */   i++;
        /* height       */   i++;
        /* ballast      */   i++;
        /* future       */   i++;
        /* input watts  */   i++;

        need(i + (size_t) n_vert + n_horiz + (size_t) n_vert * n_horiz);
        m_vert.resize(n_vert);
        for (int k = 0; k < n_vert; ++k) m_vert[k] = (ScalarFloat) n[i++];   // degrees
        m_horiz.resize(n_horiz);
        for (int k = 0; k < n_horiz; ++k) m_horiz[k] = (ScalarFloat) n[i++]; // degrees

        // candela[h][v]: grouped by horizontal angle, all vertical angles inner.
        m_cand.assign((size_t) n_vert * n_horiz, 0.f);
        for (int h = 0; h < n_horiz; ++h)
            for (int vv = 0; vv < n_vert; ++vv)
                m_cand[(size_t) h * n_vert + vv] = (ScalarFloat)(cand_mult * n[i++]);

        // Emission always spans the full azimuth; the table stores only a symmetry
        // wedge (LM-63: last horizontal angle 0=rotational, 90=quadrant,
        // 180=bilateral, 360=full). lookup_cd() folds the query angle into it.
        m_theta_max = dr::deg_to_rad(m_vert.back());        // radians
        m_symmetric = (n_horiz == 1);
        m_phi_min   = 0.f;
        m_phi_max   = (ScalarFloat)(2.0 * M_PI);
        build_grid();

        Log(Info, "IESLight: '%s' — %d vert x %d horiz angles, theta_max=%.1f deg, "
                  "peak %.1f cd, %s",
            path.filename().string(), n_vert, n_horiz, m_vert.back(),
            *std::max_element(m_cand.begin(), m_cand.end()),
            m_symmetric ? "rotationally symmetric" : "asymmetric");
    }

    // Bilinear lookup of candela at (theta_deg, phi_deg) on the irregular table.
    ScalarFloat lookup_cd(ScalarFloat theta_deg, ScalarFloat phi_deg) const {
        auto interp = [](const std::vector<ScalarFloat> &xs, ScalarFloat x,
                         int &i0, ScalarFloat &w) {
            if (x <= xs.front()) { i0 = 0; w = 0.f; return; }
            if (x >= xs.back())  { i0 = (int) xs.size() - 2; w = 1.f; return; }
            int k = 0; while (xs[k + 1] < x) ++k;
            i0 = k; w = (x - xs[k]) / (xs[k + 1] - xs[k]);
        };
        int vi; ScalarFloat vw;
        interp(m_vert, theta_deg, vi, vw);
        auto row = [&](int h) {
            return (1 - vw) * m_cand[(size_t) h * m_vert.size() + vi] +
                        vw  * m_cand[(size_t) h * m_vert.size() + vi + 1];
        };
        if (m_symmetric) return row(0);
        int hi; ScalarFloat hw;
        interp(m_horiz, fold_phi(phi_deg), hi, hw);
        return (1 - hw) * row(hi) + hw * row(hi + 1);
    }

    // Fold an azimuth [0,360) into the tabulated horizontal wedge per LM-63
    // symmetry (deduced from the last horizontal angle).
    ScalarFloat fold_phi(ScalarFloat phi) const {
        ScalarFloat last = m_horiz.back();
        phi = std::fmod(phi, 360.f);
        if (phi < 0.f) phi += 360.f;
        if (last <= 90.f + 1e-3f) {          // quadrant symmetry
            if (phi > 180.f) phi = 360.f - phi;
            if (phi >  90.f) phi = 180.f - phi;
        } else if (last <= 180.f + 1e-3f) {  // bilateral symmetry
            if (phi > 180.f) phi = 360.f - phi;
        }                                    // else full 360: no fold
        return phi;
    }

    // Resample onto a regular (phi,theta) grid; the flux-measure density
    // I*sin(theta) drives importance sampling, and the raw I grid is kept on the
    // device for evaluating the emitted weight.
    void build_grid() {
        m_nt = 180;
        m_np = m_symmetric ? 2u : 128u;
        std::vector<ScalarFloat> flux(m_nt * m_np), inten(m_nt * m_np);
        for (uint32_t y = 0; y < m_nt; ++y) {
            ScalarFloat theta = m_theta_max * (ScalarFloat) y / (m_nt - 1);
            ScalarFloat st    = dr::sin(theta);
            ScalarFloat tdeg  = dr::rad_to_deg(theta);
            for (uint32_t x = 0; x < m_np; ++x) {
                ScalarFloat phi = m_phi_min +
                    (m_phi_max - m_phi_min) * (ScalarFloat) x / (m_np - 1);
                ScalarFloat I = lookup_cd(tdeg, dr::rad_to_deg(phi));
                inten[y * m_np + x] = I;
                flux [y * m_np + x] = I * st;
            }
        }
        m_distr = Warp(flux.data(), ScalarVector2u(m_np, m_nt));
        m_inten = dr::load<FloatStorage>(inten.data(), m_nt * m_np);
    }

    // Bilinear interpolation of the raw intensity grid at unit-square (u,v).
    Float eval_I(const Point2f &uv, Mask active) const {
        Float fx = uv.x() * (m_np - 1), fy = uv.y() * (m_nt - 1);
        UInt32 x0 = UInt32(dr::floor(fx)), y0 = UInt32(dr::floor(fy));
        x0 = dr::minimum(x0, m_np - 2); y0 = dr::minimum(y0, m_nt - 2);
        Float wx = fx - Float(x0), wy = fy - Float(y0);
        auto g = [&](UInt32 xx, UInt32 yy) {
            return dr::gather<Float>(m_inten, yy * m_np + xx, active);
        };
        return dr::lerp(dr::lerp(g(x0, y0),     g(x0 + 1, y0),     wx),
                        dr::lerp(g(x0, y0 + 1), g(x0 + 1, y0 + 1), wx), wy);
    }

    // Local direction from (u,v): nadir (theta=0) along local -Z.
    Vector3f uv_to_local(const Point2f &uv, Float &sin_theta) const {
        Float theta = m_theta_max * uv.y();
        Float phi   = m_phi_min + (m_phi_max - m_phi_min) * uv.x();
        Float st, ct = dr::cos(theta); st = dr::sin(theta);
        sin_theta = st;
        auto [sp, cp] = dr::sincos(phi);
        return Vector3f(st * cp, st * sp, -ct);
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("scale",    m_scale,    ParamFlags::Differentiable);
        cb->put("to_world", m_to_world, ParamFlags::NonDifferentiable);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f & /*dir_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [uv, pdf_unit] = m_distr.sample(spatial_sample, nullptr, active);
        Float sin_theta;
        Vector3f local_dir = uv_to_local(uv, sin_theta);

        // Unit-square pdf -> solid-angle pdf. Parameter area = phi_range*theta_max;
        // dOmega = sin(theta) dtheta dphi.
        Float phi_range = Float(m_phi_max - m_phi_min);
        Float pdf_omega = pdf_unit / (phi_range * m_theta_max *
                                      dr::maximum(sin_theta, 1e-6f));

        auto si = dr::zeros<SurfaceInteraction3f>();
        si.time = time;
        si.p    = m_to_world.value().translation();
        auto [wavelengths, spec_weight] =
            sample_wavelengths(si, wavelength_sample, active);

        Float I = eval_I(uv, active);
        Float weight = dr::select(active && pdf_omega > 0.f,
                                  m_scale * I / pdf_omega, 0.f);

        return { Ray3f(si.p, m_to_world.value() * local_dir, time, wavelengths),
                 depolarizer<Spectrum>(spec_weight) * weight };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f & /*sample*/,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        DirectionSample3f ds;
        ds.p       = m_to_world.value().translation();
        ds.n       = 0.f;
        ds.uv      = 0.f;
        ds.pdf     = 1.f;
        ds.time    = it.time;
        ds.delta   = true;
        ds.emitter = this;
        ds.d       = ds.p - it.p;
        ds.dist    = dr::norm(ds.d);
        Float inv_dist = dr::rcp(ds.dist);
        ds.d      *= inv_dist;

        Vector3f local_d = m_to_world.value().inverse() * -ds.d;
        auto [uv, ok] = local_to_uv(local_d);
        active &= ok;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.time        = it.time;
        si.wavelengths = it.wavelengths;
        si.p           = ds.p;
        auto spec = m_intensity_spec(si, active);

        Float I = eval_I(uv, active);
        Spectrum val = depolarizer<Spectrum>(spec) *
                       (m_scale * I * dr::square(inv_dist));
        return { ds, val & active };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &,
                        Mask) const override {
        return 0.f;
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f & /*sample*/,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);
        Vector3f dir = m_to_world.value() * ScalarVector3f(0.f, 0.f, -1.f);
        PositionSample3f ps(m_to_world.value().translation(), dir,
                            Point2f(0.5f), time, 1.f, true);
        return { ps, Float(1.f) };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        // Uniform "white" photometric emitter: sample wavelengths uniformly and
        // return unit weight (the candela table carries all the magnitude).
        return { dr::zeros<Wavelength>() + si.wavelengths,
                 Spectrum(1.f) & active };
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override {
        return 0.f;
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IESLight[" << std::endl
            << "  to_world = " << string::indent(m_to_world) << "," << std::endl
            << "  vert_angles = " << m_vert.size() << ","  << std::endl
            << "  horiz_angles = " << m_horiz.size() << "," << std::endl
            << "  theta_max = " << dr::rad_to_deg(m_theta_max) << " deg," << std::endl
            << "  scale = " << m_scale << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(IESLight)
private:
    // Photometric "white" spectrum weight (unit); kept as a helper for symmetry
    // with textured emitters and easy future spectral extension.
    Spectrum m_intensity_spec(const SurfaceInteraction3f &, Mask active) const {
        return Spectrum(1.f) & active;
    }

    // Inverse of uv_to_local for eval: local_dir -> (u,v) plus a validity mask.
    std::pair<Point2f, Mask> local_to_uv(const Vector3f &d) const {
        Float theta = dr::acos(dr::clip(-d.z(), -1.f, 1.f));
        Float phi   = dr::atan2(d.y(), d.x());
        phi = dr::select(phi < 0.f, phi + Float(2.0 * M_PI), phi);
        Float v = theta / m_theta_max;
        Float phi_range = Float(m_phi_max - m_phi_min);
        Float u = m_symmetric ? Float(0.5f) : (phi - Float(m_phi_min)) / phi_range;
        Mask ok = theta <= m_theta_max;
        return { Point2f(u, dr::clip(v, 0.f, 1.f)), ok };
    }

    ScalarFloat m_scale;
    std::vector<ScalarFloat> m_vert, m_horiz, m_cand;   // parsed table (host)
    ScalarFloat m_theta_max, m_phi_min, m_phi_max;
    bool m_symmetric = true;
    uint32_t m_nt = 0, m_np = 0;
    Warp m_distr;                 // importance sampling over I*sin(theta)
    FloatStorage m_inten;         // raw intensity grid (device), for the weight
};

MI_EXPORT_PLUGIN(IESLight)
NAMESPACE_END(mitsuba)
