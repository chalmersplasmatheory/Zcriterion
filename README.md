# Analytical criterion for significant runaway electron generation

This package collects some functions useful for evaluating the criterion for significant runaway electron generation in activated tokamaks described an article by B. Zaar et al (submitted for publication in JPP). These functions include the evaluation of the tritium decay and Compton scattering seeds, and the avalanche gain factor (see `runaway.py`), as well as nesessary plasma propertiers, such as the Spitzer conductivity, relevant collision frequencies, critical electric field etc (see `plasma.py`). Dreicer and hot-tail seeds will be implemented shortly. 

Also included is a module for self-consistently evaluating the electron temperature and the ion charge state distribution for a given Ohmic current, see `atomicPhysics.py`. The atomic data is obtained from [OPEN-ADAS](https://www.adas.ac.uk/openadas.php). It may be necessary to download additional data depending on which impurities are to be considered.

The criterion exists in two versions, an analytical version that can evaluated immediately, and a semi-analytical version that requires a few numerical integration. For a detailed description to the expressions implemented in each version, see the source code or the article cited above. The criterion is defined as

$$
\mathcal{Z} = N_\mathrm{ava} + \ln{n_\mathrm{seed}} > 0,
$$

where $n_\mathrm{seed}$ is the total (normalized) seed density, and $\exp{N_\mathrm{ava}}$ is the avalanche gain factor. If you have any questions, feel free to reach out.
