# chisqnet

Code used for the paper Using machine learning to auto-tune chi-squared tests for gravitational wave searches.

Uses [PyCBC](https://github.com/gwastro/pycbc) and [tensorflow](https://github.com/tensorflow/tensorflow) to train new chisq tests for the PyCBC search. Relies heavily on these two packages and contains some edited functions from [PyCBC](https://github.com/gwastro/pycbc) for prerocessing strain data.

Can be used within a PyCBC search using a modifed version of the PyCBC code [in this fork](https://github.com/connor-mcisaac/pycbc/tree/chisqnet).
