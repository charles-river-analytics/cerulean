
import pytest

from models import dimensions


def test_dimension_create():
    dims = dimensions.Dimensions(("A", 3), ("b", 14))
    assert dims.get_factor_spec() == ("Ab", (3, 14))


