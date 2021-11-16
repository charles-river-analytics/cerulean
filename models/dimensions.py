
import mypy


class Dimensions:
    """
    Base class from which other Dimensions inherit. Holds the names of entities 
    entity and the corresponding cardinality of their discrete supports. 
    This class should not be used directly, instead use `VariableDimensions` or 
    `FactorDimensions`. 
    """

    def __init__(self, *variables: tuple[str, int]):
        self.variables = variables

    def get_variables(self,) -> tuple[str]:
        """
        Returns a tuple of variables associated with the Dimensions.
        """
        return tuple(v[0] for v in self.variables)

    def get_dimensions(self,) -> tuple[int]:
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

    def get_factor_spec(self,) -> tuple[str,tuple[int,...]]:
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

    def __init__(self, var: str, dim: int):
        """
        Takes a single string variable name and single integer representing that 
        variable's support set cardinality. 
        """
        super().__init__((var, dim))

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

