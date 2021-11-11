Installation
============
The PROP model project requires Python 3.9.  It uses `conda <https://docs.conda.io/en/latest/>`_ 
as its build system.

The first step is to pull down *Overmind* from its repository or to unarchive it into a :code:`overmind` repository.  Then 
conda can be used to set up a separate environment for building and running *Overmind* applications.  Note that the 
:literal:`conda-forge` must be added to conda.

From the root of the project

.. code-block:: bash

    $ conda config --add channels conda-forge 
    $ conda create --name prop-model python=3.9
    $ conda activate prop-model
    $ python -m pip install -r requirements.txt

