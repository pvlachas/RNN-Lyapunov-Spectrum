# RNN-Lyapunov-Spectrum

This project contains an implementation of an RNN architecture trained with Backpropagation through time (BPTT) to calculate the Lyapunov Spectrum from time-series data of a dynamical system. A Gated Recurrent Unit (GRU) cell is utilized.


## Requirements (installation with venv)

The code requirements are:
- python 3.7.3
- pytorch (1.5.1)
- matplotlib, psutil, scipy, tqdm

The packages can be installed as follows: you can create a virtual environment in Python3 with:
```
python3 -m venv venv-RNN-Lyapunov-Spectrum

```
Then activate the virtual environment:
```
source ./venv-RNN-Lyapunov-Spectrum/bin/activate
```
Install the required packages with:
```
pip3 install torch matplotlib scipy psutil tqdm
```
The code is ready to run.
In the following you can test the code on the identification of the Lyapunov exponents of the three dimensional Lorenz system.

## Requirements (installation with virtualenv)

The code to get the exact environment used in the paper is:
```
pip install virtualenv
virtualenv venv-RNN-Lyapunov-Spectrum --python=python3.7.3
source ./venv-RNN-Lyapunov-Spectrum/bin/activate
pip3 install -r requirements.txt
```

## Datasets

The data to run a small demo are provided in the local ./Data folder


## Demo

In order to run the demo in a local cluster, you can navigate to the Experiments folder, and select your desired application, e.g. Lorenz3D. First, a statefull GRU-RNN is trained by running the script 0_TRAIN_RNN.sh:
```
cd ./Experiments/Lorenz3D
bash 0_TRAIN_RNN.sh
```
You can then navigate to the folder ./Results/Lorenz3D and check the different outputs of each model.
After having trained the model, the Laypunov exponents are calculated with:
```
cd ./Experiments/Lorenz3D
bash 1_CALCULATE_LYAPUNON_SPECTRUM.sh
```
At the terminal output, you can observe the progress of the LE calculation.
The code generates a plot of the calculated spectrum in the /Results folder, and prints the estimated values at the terminal output.
The iterative_prediction_length determining the length of the trajectory utilized to identify the LE, needs to be large in order to have an accurate estimation of the spectrum.
In the Lorenz (3 dimensional system) case, we set the iterative prediction length to 20000 (the rest of hyperparameters can be found in the demo script) and we get the Lyapunov exponents:
```
Exponents are:
[-1.4312247e+01 -4.3020570e-03  9.0752620e-01]
```
Indeed, these exponents are very close to the groundtruth, i.e. `(−14.5723, 0, 0.9056)` (see [reference](https://aip.scitation.org/doi/10.1063/1.4927643)).
More accurate results can be attained by averaging over many trajectories, by increasing the iterative prediction length, or tuning the hyperparameters, or training a larger RNN model (make sure that it does not overfit though!).

## Contact info

This code was developed in the [CSE-lab](https://www.cse-lab.ethz.ch).
For questions or to get in contact please refer to pvlachas@ethz.ch.

## Acknowledgments

This is joint work with:
- Jaideep Pathak ([website](http://physics.umd.edu/~jpathak/), [scholar](https://scholar.google.com/citations?user=cevw0gkAAAAJ&hl=en)) 
- Brian R. Hunt ([website](http://www.math.umd.edu/~bhunt/), [scholar](https://scholar.google.com/citations?user=ten7UlMAAAAJ&hl=en))
- Themis Sapsis ([website](http://sandlab.mit.edu/), [scholar](https://scholar.google.com/citations?user=QSPXIAQAAAAJ&hl=en))
- Michelle Girvan ([website](https://sites.google.com/umd.edu/networks/home), [scholar](https://scholar.google.com/citations?user=npKBI-oAAAAJ&hl=el)) 
- Edward Ott ([website](https://umdphysics.umd.edu/people/faculty/current/item/380-edott.html), [scholar](https://scholar.google.com/citations?user=z7boxkkAAAAJ&hl=en))
- Petros Koumoutsakos ([website](https://www.cse-lab.ethz.ch/member/petros-koumoutsakos/), [scholar](https://scholar.google.ch/citations?user=IaDP3mkAAAAJ&hl=el&oi=ao)) 

## Relevant Publications

[1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., *Backpropagation algorithms and
Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal
dynamics.* Neural Networks (2020), doi: https://doi.org/10.1016/j.neunet.2020.02.016.

[2] J. Pathak, B.R. Hunt, M. Girvan, Z. Lu, and E. Ott, *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach*, Physical Review Letters 120 (2), 024102, 2018

[3] P.R. Vlachas, W. Byeon, Z.Y. Wan, T.P. Sapsis, and P. Koumoutsakos *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, 
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018
