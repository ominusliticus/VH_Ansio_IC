## VH_Ansio_IC

Viscous hydrodynamics for anisotropic initial conditions.
Please see (in progress) for derivations and descriptions of the here-in contained methods.

Most of the code has been adapted from the C++ [implementation](https://github.com/mjmcnelis/cpu_vah.git) by Mike McNelis.
The relevant papers are: 
1. D. Bazow, U. Heinz and M. Strickland, Comput. Phys. Commun. 225 (2018) 92-113
1. M. McNelis, D. Bazow and U. Heinz, Phys. Rev. C 97, 054912 (2018)


The jupyter notebook `Anisotropic_first_order_numerics.ipynb` does the numerical evolution of the hydrodynamic equations, and outputs the energy density as a function of time (still deciding how I want to do this).

The notebook requires the files `hydro_simulation.py` and `hydro_model.py` to run. The code was packaged into these modules so the Jupyter notebook itself is more readable.