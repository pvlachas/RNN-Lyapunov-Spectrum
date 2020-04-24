# RNN-Lyapunov-Spectrum

This project contains minimal implementation of the RNN architecture trained with Backpropagation through time (BPTT) to calculate the Lyapunov Spectrum from time-series data of a dynamical system. A GRU cell is utilized.

## REQUIREMENTS

The code requirements are:
- python 3.7.3
- matplotlib, psutil, scipy, tqdm

The packages can be installed as follows: you can create a virtual environment in Python3 with:
```
python3 -m venv venv-RNN-Lyapunov-Spectrum

```
Then activate the virtual environment:
```
source venv-RNN-Lyapunov-Spectrum/bin/activate
```
Install the required packages with:
```
pip3 install torch matplotlib scipy psutil tqdm
```
The code is ready to run.
In the following you can test the code on the identification of the Lyapunov exponents of the three dimensional Lorenz system.

## DATASETS

The data to run a small demo are provided in the local ./Data folder

## DEMO

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
The iterative_prediction_length determining the length of the trajectory utilized to identify the LE, needs to be large in order to have an accurate estimation of the spectrum.
The code generates a plot of the calculated spectrum in the /Results folder, and print the estimated values at the terminal output.
In the Lorenz case, we get an estimate of 0.9080213, which is really close to the true one (approximately 0.9056).


## Note

This is only a minimal version of the code under development in the CSE-lab.
Please contact pvlachas@ethz.ch if you want to get informed, take a look at the latest version, with more features, models and capabilities.

### Relevant Publications
[1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., *Backpropagation algorithms and
Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal
dynamics.* Neural Networks (2020), doi: https://doi.org/10.1016/j.neunet.2020.02.016.

[2] *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach*, Jaideep Pathak, Brian Hunt, Michelle Girvan, Zhixin Lu, and Edward Ott
Physical Review Letters 120 (2), 024102, 2018

[3] *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, Pantelis R. Vlachas, Wonmin Byeon, Zhong Y. Wan, Themistoklis P. Sapsis and Petros Koumoutsakos
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018
   




