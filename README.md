# MSM-MLanalysis

**Under contruction :(**

The MSM-analysis workflow takes as input a "trajectory" and applies a MSM routine to extract the relevant dynamic states (whithin the markovian theoretical framework).
This repository contains tools and utils to apply the a MSM type of analysis on the data coming from MD simulations.

Handling of the MD data (using the ASE-atom python package) and the subsequent processing of such data(_i.e._, computing of atom centered descriptors) are briefly discussed in two py-notebooks:
-   `handling_trajs_ASEtools.ipynb`
-   `compute_soap_descriptors.ipynb`

The MSM is built by scanning a series of lag-times (_e.g._ as defined in the `.json` file in the `lag_time_scan` section) and it involves a cyclic worflow that can be summarised as:
1.  definition of the given $\tau^*$ (lag time).
2.  using the tICA dimensionality reduction on the MD data (in this case processed as descriptor time-serie).
3.  the tICA space is discretised in many domains, _e.g._, first guess on the discrete dynamic domains (the trajectory becomes the time evolution of these labels)
4.  a first MSM is built on the time-evolution of these discretised states
5.  PCCA+ is employed to merge togheters a number of dynamic states (defined by either `eigenval_treshold` or `fixed_eigensates` in the `.json`)

To run such cycle on given descriptor files one can use the command:
```bash
python msmanalysis.py -c msm_confing.json
```

The information gathered by the MSM cycle is used to get the optimal lag-time $\tau^*$ and the corresponding amount of CG-dynamic states (_i.e._ PCCA+ reduced states).