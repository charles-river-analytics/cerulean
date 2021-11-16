
import pytest

from models import dimensions


def test_dimension_create():
    dims = dimensions.Dimensions(("A", 3), ("b", 14))
    assert dims.get_factor_spec() == ("Ab", (3, 14))

    dims = dimensions.Dimensions(("A", 3), ("b", 14), ("q", 5))
    assert dims.get_factor_spec() == ("Abq", (3, 14, 5))


def test_variable_dimension():
    vd = dimensions.VariableDimensions("A", 5)
    assert vd.get_factor_spec() == ("A", (5,))

    with pytest.raises(TypeError):
        dimensions.VariableDimensions(("A", 5))


def test_factor_dimensions():
    a = dimensions.VariableDimensions("a", 2)
    b = dimensions.VariableDimensions("b", 3)
    c = dimensions.VariableDimensions("c", 4)

    a_ = dimensions.FactorDimensions(a)
    ab = dimensions.FactorDimensions(a, b)
    abc = dimensions.FactorDimensions(a, b, c)

    assert a_.get_factor_spec() == ("a", (2,))
    assert ab.get_factor_spec() == ("ab", (2, 3))
    assert abc.get_factor_spec() == ("abc", (2, 3, 4))
    # NOTE: order matters!!!
    abcca = dimensions.FactorDimensions(a, b, c, c, a)
    assert abcca.get_factor_spec() == ("abcca", (2, 3, 4, 4, 2))
    acbca = dimensions.FactorDimensions(a, c, b, c, a)
    assert acbca.get_factor_spec() == ("acbca", (2, 4, 3, 4, 2))
