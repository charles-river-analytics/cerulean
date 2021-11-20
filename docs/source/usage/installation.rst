Installation
============
`cerulean` requires Python 3.9.  It uses `conda <https://docs.conda.io/en/latest/>`_ 
as its build system.

Here's how to set up from the root of the project. Note that we've named the virtual 
environment `proper-fm` after the parent project under which `cerulean` was developed.

.. code-block:: bash

    $ conda config --add channels conda-forge 
    $ conda create --name proper-fm python=3.9
    $ conda activate proper-fm
    $ python -m pip install -r requirements.txt

