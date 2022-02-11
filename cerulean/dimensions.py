
from typing import Mapping, Optional, Union, Tuple

import mypy
import numpy as np

from . import transform


def _mapping_invert(m: Mapping) -> dict:
    return {v : k for (k, v) in m.items()}


class Dimensions:
    """
    Base class from which other Dimensions inherit. Holds the names of entities 
    entity and the corresponding cardinality of their discrete supports. 
    This class should not be used directly, instead use `VariableDimensions` or 
    `FactorDimensions`. 
    """

    def __init__(self, *variables: Tuple[str, int]):
        if len(variables) < 1:
            raise ValueError("Must pass at least one (var, dim) tuple.")
        self.variables = variables

    def __len__(self,):
        return len(self.variables)

    def get_variables(self,) -> tuple[str,...]:
        """
        Returns a tuple of variables associated with the Dimensions.
        """
        return tuple(v[0] for v in self.variables)

    def get_dimensions(self,) -> tuple[int,...]:
        """
        Returns a tuple of integers, each of which is the cardinality of one dimension
        of the support set.
        """
        return tuple(v[1] for v in self.variables)

    def get_variable_str(self,) -> str:
        """
        Returns a single string that is the concatenation of all component names.
        E.g., if this Dimensions relates "a", "b", and "d", this method would return 
        "abd".
        """
        return "".join(self.get_variables())

    def get_factor_spec(self,) -> Tuple[str,Tuple[int,...]]:
        """
        Returns a representation of the Dimensions that's useful for the functional
        factor graph interpretation. 
        If this dimensions object relates "a", "b", and "d" with respective support 
        cardinalities 2, 3, and 4, this method would return ("abd", (2, 3, 4)).
        """
        return (self.get_variable_str(), self.get_dimensions())


class VariableDimensions(Dimensions):
    """
    A dimensions that takes only one name and one integer representing the 
    cardinality of the variable's support set.
    """

    def __init__(self, var: str, dim: int, bins: Optional[np.ndarray]=None,):
        """
        Takes a single string variable name and single integer representing that 
        variable's support set cardinality. 
        """
        super().__init__((var, dim))
        self.bins = bins

    def __str__(self,):
        if type(self.bins) is not None:
            num_bin_edges = len(self.bins)
            _min = round(self.bins[0], 3)
            _max = round(self.bins[-1], 3)
            return f"VariableDimensions({self.variables}, bins (number = {num_bin_edges - 1}) are [{_min},...,{_max}])"
        else:
            return f"VariableDimensions({self.variables})"

    def __getitem__(self, value):
        if type(self.bins) is None:
            raise TypeError("Can't convert value with nonexistent bins!")
        if value >= self.bins[-1]:
            return len(self.bins) - 1
        elif value <= self.bins[0]:
            return 0
        else:
            return np.digitize(value, self.bins, right=True)

    def set_bins(self, bins):
        assert type(bins) is np.ndarray
        self.bins = bins

    def transform_bins(self, transform_fn, base_value,):
        """
        Transforms bins in place using the `transform` and a `base_value`. 
        The bins are transformed as `bins <- transform(bins, base_value)`.
        """
        if type(self.bins) is not None:
            self.bins = transform._stationary_to_nonstationary_bins(transform_fn, base_value, self.bins)

    def __repr__(self,):
        return f"VariableDimensions({self.variables})"


class FactorDimensions(Dimensions):
    """
    A Dimensions that relates multiple VariableDimensions.
    """

    def __init__(self, *variables: VariableDimensions):
        """
        Takes an arbitrary number of `VariableDimensions`.
        """
        new_variables = [
            (v.get_variable_str(), v.get_dimensions()[0]) for v in variables
        ]
        super().__init__(*new_variables)

    def __repr__(self,):
        return f"FactorDimensions({self.variables})"


class DimensionsFactory:
    """
    Helper for creating `VariableDimensions` and `FactorDimensions`. 

    Example usage::
    
        # create a factory
        factory = DimensionFactory("my variable", "my other variable")

        # actually register the variables and their dimensions
        factory("my variable", 10)  # has 10 cutpoints
        factory("my other variable", 29)  # has 29 cutpoints

        # make some factor dimensions
        fd1 = factory(("my variable",))  # dimensions for factor of degree 1
        fd2 = factory(("my variable", "my other variable"))  # dimensions for factor of degree 2
    
    Calling this object raises `ValueError` if variable names don't exist.
    """

    def __init__(self, *variable_names: str):
        self.names2strings = transform.get_names2strings(*variable_names)
        self.names2variables = dict()

    def __call__(
        self,
        names: Union[str,tuple[str,...]],
        n_cutpoints: Optional[int]=None,
        bins: Optional[np.ndarray]=None,
    ) -> Union[VariableDimensions, FactorDimensions]:
        if type(names) is str:
            if n_cutpoints is None:
                raise ValueError("Must pass n_cutpoints for variable dimension generation!")
            vd = VariableDimensions(self.names2strings[names], n_cutpoints, bins=bins,)
            self.names2variables[names] = vd

        elif type(names) is tuple:
            if n_cutpoints is not None:
                raise ValueError("n_cutpoints already defined by variables!")
            if not all((x in self.names2strings.keys() for x in names)):
                raise ValueError("At least one of the names isn't associated with a variable!")
            return FactorDimensions(*(self.names2variables[name] for name in names))

    def mapping(self,) -> dict[str, str]:
        """
        Returns the mapping from variable names to their string representations used 
        by `opt_einsum`.
        """
        return self.names2strings

    def mapping_inverse(self,) -> dict[str, str]:
        """
        Returns the mapping from string representations used by `opt_einsum` to variable names.
        """
        return _mapping_invert(self.names2strings)

    def get_variable(self, name: str) -> VariableDimensions:
        """
        Returns the `VariableDimension` associated with `name`.
        """
        return self.names2variables[name]
