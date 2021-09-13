# SeepagePINN -Investigating Groundwater Flows using Physics Informed Neural Networks
Oden Institute for Computational Engineering and Sciences / Jackson School of Geosciences / University of Texas Institute for Geophysics
The University of Texas at Austin

SeepagePINN is a technique to predict model parameters such as hydraulic conductivity and free surface profiles for groundwater flows by using Dupuit - Boussinesq and Di Nucci approximations to training the model.


## Getting Started

### Overview

The ground water flow PINN technique utilizes the information from the known (training) data and the underlying physics from either the classical Dupuit-Boussinesq approximation or more recent DiNucci model. The effect of higher order vertical flows on the overall groundwater flow dynamics is investigated. The data is obtained from steady-state analytical results and laboratory experiments in figure (a).

SeepagePINN has also been used to invert for model parameters such as hydraulic conductivity, in addition to predicting free surface profiles directly from the training data and physics models in figure (b).

![cover](src/paper/Cover.pdf?raw=true)


### Dependences

SeepagePINN requires the following packages to function:
- [Python](https://www.python.org/) version 3.5+
- [h5py](http://www.h5py.org/) >= 3.3.0
- [Numpy](http://www.numpy.org/) >= 1.16
- [scipy](https://www.scipy.org/) >=1.5
- [argparse](https://pypi.org/project/argparse/) >= 1.4.0
- [*pandas*](https://pandas.pydata.org/) >= 1.3.1
- [TensorFlow](https://www.tensorflow.org/) 0.10.0rc0, also tested with
  TensorFlow = 1.x

If you want to train the 1D Unsteady Groundwater Flow Model with Di Nucci approximation, then we also need to install:
- [fenics](https://fenicsproject.org/) 

### Numerical Data Sources
Our SeepagePINN Model train, validation and test datasets by Dupuit-Boussinesq and Di Nucci model.


### Experimental Data Sources
For training the experimental data, we need to define X, u, L, W, K parameters.
X: space domain
u: training solution (free surface)
L: length
W: 
K: hydraulic conductivity

### Running seepagePINN
(by importing argparse in python code)
- python experimental_invert.py --help

usage: experimental_invert.py [-h] [-c CASE] [-n N_EPOCH]
                              [-m {dinucci,dupuit}] [-r]

Select PDE model

optional arguments:
  -h, --help            show this help message and exit
  -c CASE, --case CASE  case name
  -n N_EPOCH, --N_epoch N_EPOCH
                        Number of training epochs
  -m {dinucci,dupuit}, --flow_model {dinucci,dupuit}
                        PDE choice for generating data: dinucci or dupuit
  -r, --random          Do not set constant seed


### Quick Usage


## Authors
- Mohammad Afzal Shadab
- Dingcheng Luo
- Yiran Shen
- Eric Hiatt
- Marc Andre Hesse
