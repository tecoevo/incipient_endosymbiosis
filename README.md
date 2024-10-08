### This repository contains the scripts, Mathematica notebooks, and figure generation pipelines for the paper "On the structure of an evolutionary transition: dependence and cohesion in incipient endosymbioses".

## Overview ##

This repository is organised into three folders: 

1. `FigureSources` contains the figures themselves (as .pdf files), as well as the OmniGraffle files used to make them. 
2. `SimulationCode` contains all the Python scripts for running evolutionary simulations and plotting their results. 
3. `Analytics` contains Mathematica notebooks used to perform analytical calculations related to finding fixed points, determining their stability, manipulating invasion fitnesses, etc.

Further detailed information required for reproducing the results in the paper is given below.

## Analytics ##

This folder contains 3 files, each of which is further commented within and should be self-explanatory. 

* `obligacy-evolution.nb` deals with understanding the independent evolution of the obligacies, in particular for the analytics underlying the invasion fitness.  
* `adhesion-evolution.nb` deals with understanding the independent evolution of the adhesions. 
* `obligacy-adhesion-coevolution.nb` deals with deriving the invasion fitness when both adhesions and obligacies can evolve.
* `collective-logistic-evolution.nb` deals with deriving the invasion fitness when both collective growth is logistic. 

The contents of these files share some similarities, but are made separate -- at the cost of some redundancy -- for convenience of usage.

## SimulationCode ##

This folder is organized into two top-level subfolders: `C-exp-growth` and `C-logistic-growth` which contain the relevant scripts for the models having exponential and logistic growth for the collective population.

Both these folders have 4 subfolders:

* `feasibility-stability` contains scripts for understanding the features of the ecological dynamics.
* `obligacy-evolution` contains scripts for simulating and plotting the independent evolution of the obligacies.
* `adhesion-evolution` contains scripts for simulating and plotting the independent evolution of the adhesions.
* `coevolution` contains scripts for simulating and plotting the coevolution of the obligacies with adhesions.

Note: all scripts in `./C-exp-growth/` re-compute the stochastic evolutionary trajectories every time since the exponential growth model is analytically solvable, making computations faster since we can put in some expressions 'by hand'. However, all scripts used to make figures in `./C-logistic-growth/` simply plot an already present set of data files which were obtained by running the relevant scripts on a cluster. For example, in `./C-logistic-growth/obligacy-evolution`, the file to run on a cluster is `HPC-obligacy-trajectory.py` And once all the data files from there are in `./data` (available on request, not included for their large size), one can use `analyse-obligacy-trajectory-data.py` to plot them. 

## FigureSources ##

All these figures have corresponding OmniGraffle files which can be found by replacing the ".pdf" extension in their names with ".graffle". When mentioned, data and plots can be found in the folders `./results/data` or `./results/plots` of the appropriate folder. For example, in Figure 2 of the main text (`logistic-collective-evolution.pdf`), the appropriate directories are `SimulationCode/C-logistic-growth/obligacy-evolution/` and `SimulationCode/C-logistic-growth/adhesion-evolution/`. The data for making the plots in these figures is stored under the jobname specified by the parameters 'a=0.1_rHS=10.0_CHS=250.0' i.e. for panels (a,c) the data and plots are in is in `SimulationCode/C-logistic-growth/obligacy-evolution/results/data(or plots)/a=0.1_rHS=10.0_CHS=250.0` and for panels (b,d) the data and plots are in `SimulationCode/C-logistic-growth/adhesion-evolution/results/data(or plots)/a=0.1_rHS=10.0_CHS=250.0`. In the following we will specify only jobname with the understanding that this uniquely specifies a directory in the appropriate version of `results/data` or or `results/plots`.

* Figure 1 (`model-concept.pdf`) is entirely conceptual, and does not make use of plots generated by any scripts.
* Figure 2 (`logistic-collective-evolution.pdf`) requires the output of scripts in `SimulationCode/C-logistic-growth/obligacy-evolution/obligacy-trajectory.py`,  `SimulationCode/C-logistic-growth/adhesion-evolution/adhesion-trajectory.py`, and `SimulationCode/C-logistic-growth/feasibility-stability/simulate-popdyn.py`. Panels (a,b) are from `obligacy-trajectory.py`; panels (d,e) are from `adhesion-trajectory.py`, and panels (c,f) are from simulate-popdyn.py run for each value along the evolutionary trajectory generated in the other panels. Data and plots under jobname `a=0.1_rHS=10.0_CHS=250.0`. 
* Figure 3 (`vary-collective-parameters.pdf`) requires 2 calls each of `SimulationCode/C-logistic-growth/obligacy-evolution/obligacy-trajectory.py` and `SimulationCode/C-logistic-growth/adhesion-evolution/adhesion-trajectory.py`, with appropriate parameter values each. Data and plots under jobnames 'a=0.1_rHS=**_CHS=**' where ** stands for rHS and CHS values appropriate to the panel in question.
* Figure 4 (`exponential-model-independent-evolution.pdf`) requires the output of `SimulationCode/C-exp-growth/obligacy-evolution/obligacy-trajectory.py` (panels a,b) and `SimulationCode/C-exp-growth/adhesion-evolution/adhesion-trajectory.py` (panels c,d). Data and plots not stored since they are very quickly generated. 
* Figure 5 (`dependence-cohesion-coevolution-exponential.pdf`) requires the output of `SimulationCode/C-exp-growth/coevolution/coevolution-trajectory.py`. Data and plots not stored since they are very quickly generated.
 
* Supplementary Figure S.1 (`eqpopsizes-logistic-varyingomegas.pdf`) requires the output of `C-logistic-growth/feasibility-stability/popsize-trait-variation.py` with parameter 'which = omega`. 
* Supplementary Figure S.2 (`eqpopsizes-logistic-varyingalphas.pdf`) requires the output of `C-logistic-growth/feasibility-stability/popsize-trait-variation.py` with parameter 'which = alpha`
* Supplementary Figure S.3 (`exponential-stability-numerics.pdf`) requires the output of `C-exp-growth/feasibility-stability/stability-interior-fpt.py`
* Supplementary Figure S.4 (`adhesionevolution-logistic-HSidentical.pdf`) requires the output of `C-logistic-growth/adhesion-evolution/adhesion-trajectory.py` with appropriate parameters. Data and plots under jobnames 'a=0.1_rHS=40.0_CHS=500.0_HS-identical'.
* Supplementary Figure S.5 (`testing-KC-cutoff.pdf`) requires the output of `C-logistic-growth/obligacy-evolution/obligacy-trajectory.py`. Data and plots under jobnames 'a=5_rHS=40.0_CHS=500.0' and 'a=0.01_rHS=40.0_CHS=10.0'.
* Supplementary Figure S.6 (`alphas-reach-11.pdf`) requires the output of `C-logistic-growth/adhesion-evolution/adhesion-trajectory.py`. Data and plots under jobname 'a=5_rHS=40.0_CHS=10000.0'.
* Supplementary Figures S.7-S.9 (`appendix-dependence-cohesion-iI/iiI/iiII.pdf`) All three figures can be made by changing the appropriate parameter which can be found in `C-exp-growth/coevolution/coevolutionary-trajectory-modified.py`.
* Supplementary Figure S.10 (`dependence-cohesion-robustness.pdf`) All panels can be made by changing the appropriate parameters in `C-exp-growth/coevolution/coevolutionary-trajectory-modified.py`.

