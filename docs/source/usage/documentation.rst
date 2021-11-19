Documentation
=============

Factor graphs
*************

This module contains functions and classes related to factor graphs. Factor graphs are undirected
bipartite graphs that relate variable nodes, which are quantities of interest such as the 
genre into which a movie may be classified or the price of a limit bid order, to factor
nodes, which are non-negative functions that relate values of variables to one another.
If a factor graph has :math:`N` variables :math:`x_1,...,x_N`, the factor graph can 
be expressed as

.. math:: p(x_1,...,x_N) = \frac{1}{Z(\theta)} 
    \prod_{\mathrm{cl}}\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \theta)
    :label: fg-eq

The notation :math:`x_\mathrm{cl}` denotes a clique of variables that are related by the factor
:math:`\psi_\mathrm{cl}`. For example, a factor graph that relates all pairs of nodes would have
joint density given by

.. math:: p(x_1,...,x_N) = \frac{1}{Z(\theta)} 
    \prod_{1\leq i < j \leq N}\psi_{i,j}(x_i, x_j; \theta)

The vector :math:`\theta` denotes the vector of parameters for the entire graph (note that
"vector" in this context can be interpreted as "all tensors of parameters"; you could create a
vector from all tensors of parameters by `squeeze`-ing each of them and `concat`-enating them 
together). The value of a factor is given by
:math:`\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \theta) = \theta_{x_\mathrm{cl}}`, the corresponding
element in the :math:`|\mathrm{cl}|` dimensional tensor of values.

Values of a factor denote how likely it is that variable elements co-occur. This module is concerned
with two factor interpretations: probability factors, which denote un-normalized probablility that
variable elements co-occur; and constraint factors, which are :math:`\{0,1\}`-valued tensors that
denote whether it is possible or not for variable elements to co-occur.
The number :math:`Z(\theta)` in Eq. :eq:`fg-eq` is a normalization constant. In the case of
a factor graph containing only constraint nodes its value is immaterial (since a configuration of nodes
is valid iff :math:`p(x_1,...,x_N)` is nonzero), but in the case of a factor graph representing a 
probability mass function (pmf) it is the normalizer, or the number that makes :math:`p(x_1,...,x_N)`
a valid pmf. In this case, the partition function is equal to

.. math:: Z(\theta) = \sum_{x_1,...,x_N}
    \prod_{\mathrm{cl}}\psi_{\mathrm{cl}}(x_{\mathrm{cl}}; \theta)

A naive computation of this quantity is exponential in the number of variables in the factor graph.
Exploiting the structure of the factor graph during computation lowers the complexity to only
exponential in the size of the highest-degree factor, which may be a significant reduction in complexity
(e.g., for a factor graph with thousands of variables but only pairwise relationships between variables).