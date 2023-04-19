# Sequential Monte Carlo Sampling for Abstract Data Types
Python classes describing sampling from distributions over
abstract data types.

Includes Markov chain Monte Carlo (MCMC) sampling, Sequential Monte Carlo (SMC) sampling
and decision forest (using sk-learn).

Published in DOI: XXXX

Copyright (c) 2023, University of Liverpool.

## Requirements

The versions of Python and Python packages used for the results in XXX
are as follows:
 - Python 3.9.16
 - numpy 1.19.2
 - scipy 1.5.2
 - scikit-learn 0.23.2
 - pandas 1.1.3
 - mpi4py 3.1.4

Either OpenMPI or Microsoft MPI should be installed to
enable running with mpi4py for distributed parallel processing.

## Running Tests

Tests can be found in `examples` with corresponding data sets.

Configuration of test cases is achieved by setting the following boolean variables (found in the example scripts):
```python
MCMC_one_to_many = False
MCMC_many_to_one = False
SMC = False
DF = False
```

The number of particles and iterations are also configured using the variables `N` and `T` respectively (also found in the example scripts).

The number of particles and number of processor cores should
be set to a power of 2 (e.g. 256, 512). The number of cores 
must be less than or equal to the number of particles,
and is specified on the command line. Example scripts can be run using MPI with the following command:
```bash
mpiexec -n <NUM_CORES> python <path/to/script.py>
```