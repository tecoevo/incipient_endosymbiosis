### This repository contains the scripts, Mathematica notebooks, and figure generation pipelines for the paper ["On the structure of an evolutionary transition: dependence and cohesion in incipient endosymbioses"](https://doi.org/10.1101/2023.07.17.549359) (Athreya, Czuppon, Gokhale).

## Overview ##

This repository is organised into three folders: 

1. `FigureSources` contains the figures themselves (as .pdf files), as well as the OmniGraffle files used to make them. 
2. `SimulationCode` contains all the Python scripts for running evolutionary simulations and plotting their results. 
3. `Analytics` contains Mathematica notebooks used to perform analytical calculations related to finding fixed points, determining their stability, manipulating invasion fitnesses, etc.

Further detailed information required for reproducing the results in the paper is given below.

## Analytics ##

This folder contains 3 files, each of which deals loosely with one of the figures of the main text. They are further commented within and should be self-explanatory. 

* `obligacy-evolution.nb` deals with understanding the independent evolution of the obligacies, in particular for the analytics underlying the invasion fitness and other insights behind Figure 1. 
* `adhesion-evolution.nb` deals with understanding the independent evolution of the adhesions, in particular for the analytics and insights behind Supplementary Figure 2. 
* `obligacy-adhesion-coevolution.nb` deals with deriving the invasion fitness when both adhesions and obligacies can evolve. 

The contents of these files share similarities, but are made separate -- at the cost of some redundancy -- for convenience of usage.

## SimulationCode ##

This folder is organized into two top-level subfolders: `C-exp-growth` and `C-logistic-growth` which contain the relevant scripts for the models having exponential and logistic growth for the collective population.

Both these folders have 4 subfolders:

* `feasibility-stability` contains scripts for understanding the features of the ecological dynamics.
* `obligacy-evolution` contains scripts for simulating and plotting the independent evolution of the obligacies.
* `adhesion-evolution` contains scripts for simulating and plotting the independent evolution of the adhesions.
* `coevolution` contains scripts for simulating and plotting the coevolution of the obligacies with adhesions.

Note: all scripts in `./C-exp-growth/` re-compute the stochastic evolutionary trajectories every time since the exponential growth model is analytically solvable, making computations faster since we can put in some expressions 'by hand'. However, all scripts used to make figures in `./C-logistic-growth/` simply plot an already present set of data files which were obtained by running the relevant scripts on a cluster. For example, in `./C-logistic-growth/obligacy-evolution`, the file to run on a cluster is `HPC-obligacy-trajectory.py` And once all the data files from there are in `./data` (available on request, not included for their large size), one can use `analyse-obligacy-trajectory-data.py` to plot them. 

## FigureSources ##

All these figures have corresponding OmniGraffle files which can be found by replacing the ".pdf" extension in their names with ".graffle". 

* Figure 1 (`model-concept.pdf`) is entirely conceptual, and does not make use of plots generated by any scripts.
* Figure 2 (`obligacy-trajectory.pdf`) requires the output of scripts in `SimulationCode/C-exp-growth/obligacy-evolution/`. Panels (a,b) are from `./obligacy-trajectory.py`; panels (c,d) are from `./feasibility-bound-variation-ad.py`.
* Figure 3 (`collective-logistic-evolution.pdf`) requires the output of SimulationCode/C-logistic-growth. Panel (a) is from `./obligacy-evolution/analyse-obligacy-trajectory-data.py`; panel (b) is from `./adhesion-evolution/analyse-adhesion-trajectory-data.py`.
* Figure 4 (`dependence-cohesion-coevolution-exponential.pdf`) requires the output of `SimulationCode/C-exp-growth/coevolution/coevolution-trajectory.py`. 

* Supplementary Figure 1 (`exponential-stability-numerics.pdf`) requires the output of `C-exp-growth/feasibility-stability/stability-interior-fpt.py`. 
* Supplementary Figures 2 (`adhesion-trajectory.pdf`) requires the output of `C-exp-growth/adhesion-evolution/adhesion-trajectory.py`
* Supplementary Figures 3-5 (`appendix-dependence-cohesion-iI/iiI/iiII.pdf`) All three figures can be made by changing the appropriate parameter which can be found in `C-exp-growth/coevolution/coevolutionary-trajectory-modified.py`.
* Supplementary Figure 6 (`dependence-cohesion-robustness.pdf`) All panels can be made by changing the appropriate parameters in `C-exp-growth/coevolution/coevolutionary-trajectory-modified.py`.
* Supplementary Figure 7 (`dependence-cohesion-logistic.pdf`) requires the results of `C-logistic-growth/coevolution/analyse-coevolution-trajectory-data.py`.

