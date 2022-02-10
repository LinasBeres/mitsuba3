from importlib import import_module as _import
import pytest

def test01_import_mitsuba_variants():
    import mitsuba as mi

    assert mi.variant() is None
    with pytest.raises(ImportError, match='Before importing '):
        mi.Float

    # Set an arbitrary variant
    variant = mi.variants()[-1]
    mi.set_variant(variant)
    assert mi.variant() == variant

    # Importing specific variant for the first time should change the current variant
    mi_var = _import(f'mitsuba.{variant}')
    assert mi_var.variant() == variant

    # Change the current variant
    mi.set_variant('scalar_rgb')
    assert mi.variant() == 'scalar_rgb'
    assert mi_var.variant() == variant


def test02_import_submodules():
    import mitsuba as mi
    mi.set_variant(mi.variants()[-1])

    # Check C++ submodules
    assert mi.math.ShadowEpsilon is not None
    assert mi.warp.square_to_uniform_disk is not None

    # Check Python submodules
    assert mi.chi2.ChiSquareTest is not None

    # Import nested submodules
    import mitsuba.test.util
    assert mi.test.util.fresolver_append_path is not None


def test03_import_from_submodules():
    import mitsuba
    mitsuba.set_variant('scalar_rgb')

    from mitsuba import Float
    assert Float == float

    from mitsuba.warp import square_to_uniform_disk
    assert square_to_uniform_disk is not None

    from mitsuba.chi2 import ChiSquareTest
    assert ChiSquareTest is not None

    from mitsuba.test.util import fresolver_append_path
    assert fresolver_append_path is not None


def test04_check_all_variants():
    import mitsuba as mi
    for v in mi.variants():
        mi.set_variant(v)


def test05_register_ad_integrators():
    import mitsuba as mi

    for variant in mi.variants():
        if not '_ad_' in variant:
            continue
        mi.set_variant(variant)
        integrator = mi.load_dict({'type': 'prb'})
        assert integrator.class_().variant() == variant
