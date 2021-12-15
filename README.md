# SeepagePINN -Investigating Groundwater Flows using Physics Informed Neural Networks
Oden Institute for Computational Engineering and Sciences / Jackson School of Geosciences / University of Texas Institute for Geophysics
The University of Texas at Austin

SeepagePINN is a technique to predict model parameters such as hydraulic conductivity and free surface profiles for groundwater flows by using Dupuit - Boussinesq and Di Nucci approximations to training the model.


## Getting Started

### Overview

The ground water flow PINN technique utilizes the information from the known (training) data and the underlying physics from either the classical Dupuit-Boussinesq approximation or more recent DiNucci model. The effect of higher order vertical flows on the overall groundwater flow dynamics is investigated. The data is obtained from steady-state analytical results and laboratory experiments in figure (a).

SeepagePINN has also been used to invert for model parameters such as hydraulic conductivity, in addition to predicting free surface profiles directly from the training data and physics models in figure (b).

![cover](/cover/cover.png?raw=true)

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

<!--
If you want to train the 1D Unsteady Groundwater Flow Model with Di Nucci approximation, then we also need to install:
- [fenics](https://fenicsproject.org/) 
-->

### Numerical Data Sources
Our SeepagePINN Model train, validation and test datasets by Dupuit-Boussinesq and Di Nucci model.


### Experimental Data Sources
For training the experimental data, we need to define X, u, L, W, K parameters.\
X: horizontal dimension (m)\
u: training solution (free surface height in m)\
L: length (m)\
W: width in the third dimension (m)\
K: hydraulic conductivity (m/s)

### Running seepagePINN
(by importing argparse in python code)
```
python experimental_all.py --help
```
<pre>
usage: experimental_all.py [-h] [-c {1mm,2mm}] [-n N_EPOCH] [-N N_TRAINING]
                           [-r] [--regularization {average,max}]

Select PDE model

optional arguments:
  -h, --help            show this help message and exit
  -c {1mm,2mm}, --case {1mm,2mm}
                        Case name
  -n N_EPOCH, --N_epoch N_EPOCH
                        Number of training epochs
  -N N_TRAINING, --N_training N_TRAINING
                        Number of training sets
  -r, --random          Do not set constant seed
  --regularization {average,max}
                        selection of regularization parameter

</pre>

### Quick Usage (MacOS)


1. Install the dependencies in a "Conda environment":

    i. Create an environment: conda create **environment name**\
    ii. Activate the environment: conda activate **environment name**\
    iii. Install the dependent libraries (given in dependencies): conda install **library name**
```
conda create -n seepage python=3.7
conda activate seepage 
conda install tensorflow==1.14
conda install matplotlib pandas scipy h5py
```
<!--
```
conda create -n seepage -c uvilla -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter python=3.7
conda activate seepage
conda install -c conda-forge tensorflow==1.13.2
conda install -c conda-forge numpy=1.16.6 -y
conda install -c conda-forge pandas -y
conda install -c anaconda scipy=1.5.3 -y
conda install -c anaconda h5py=3.3.0 -y
```
-->
2. Download the github repository and unzip the package contents or clone the repository.
```
git clone https://github.com/dc-luo/seepagePINN.git
```
3. Move to the specific folder on steady results
```
cd seepagePINN/src/steady/paper/
```
4. Run the python program in Mac terminal using experimental_all.py [-h] [-c CASE] [-n N_EPOCH] [-m {dinucci,dupuit}] [-r]
for example:
```
python experimental_all.py -c 1mm -n 20000
```
and to visualize the training results
python viz_exp.py -c 1mm -u --show
## Authors
- Mohammad Afzal Shadab
- Dingcheng Luo
- Yiran Shen
- Eric Hiatt
- Marc Andre Hesse


## References / Related publications
[1] Shadab, M. A., Luo, D., Shen, Y., Hiatt, E., & Hesse, M. A. (202X). Investigating Steady Unconfined Groundwater Flow using Physics Informed Neural Networks. Water Resources Research (in preparation).

[2] Hesse, M. A., Shadab, M. A., Luo, D., Shen, Y., & Hiatt, E. (2021). Investigating groundwater flow dynamics using physics informed neural networks (pinns). In 2021 agu fall meeting. (H34F-03).

[3] Shadab, M. A., Luo, D., Shen, Y., Hiatt, E., & Hesse, M. A. (2021). Investigating fluid drainage from the edge of a porous reservoir using physics informed neural networks. In 2021 siam annual meeting (an21). Session: CP15 - Machine Learning and Data Mining.
