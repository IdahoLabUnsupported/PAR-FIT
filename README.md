<p align="center">
  <img src="https://github.inl.gov/edward-chen/DARE/blob/main/resources/Logo_2.png" width="400">
</p>  

# Introduction
Particle Filter for Inference Testing (PAR-FIT) is a generic method that uses particles filters on a training dataset to determine if a model (whether built with neural networks or others) is represented by the dataset. The particle filter can be used for several different purposes including uncertainty quantification of model predictions, anomaly detection for time-series predictions, forecasting, individual sensor anomaly detection, and out-of-distribution detection in real-time environments where speed and efficiency are required. 

The particle filter relies on two assumptions:

1. Training data and/or operational data represent the baseline condition of the system. This data may have been used to train the model or collected during operation. Deviations from this baseline, whether occuring in a single sensor measurement or multiple, represent undesirable trends in the system that need to be detected. These deviations may stem from a variety of root causes, including but not limited to, miscalibration of sensors, sensor degradation, or human related errors.
2. The reliability of model prediction is relative to proximity to the baseline

# Documentation
This repository uses Sphinx to auto-generate documentation. Sphinx is not required to compile the documentation, pre-compiled html files can be found in PAR-FIT/docs/_built/html/. Clicking on any of the html files in this folder will provide access to the pre-compiled documentation. 

# Deployment 
PAR-FIT is a standalone evaluation model that does not need to be developed in tandem with predictive models. If the training/operational dataset is known, and the model input and output are known, then PAR-FIT can make an evaluation of model predictions. This can be used in two different manners:

1. During model development where PAR-FIT may be used to gauge what data is being used in a prediction and whether that models prediction is an interpolation or extrapolation
2. During model deployment, where PAR-FIT may be used to evaluate whether sensor(s) are anomalous or normal and the data uncertainty behind that prediction. PAR-FIT may be implemented in real-time or for post-processing purposes.  

# Quick Installation
Clone the repository into a folder of your choice using ```git clone <prompt>```.

There are four modules in ```src``` folder that are callable; ```Particle```, ```Scoring```, ```Swarm```, and ```Resample```. 

# Test Case
Navigate to ```examples/Particle_Examples```. There are two pre-built test cases that can be run immediately to ensure everything has been downloaded appropriately. 

The first test case is labelled ```Transient_007_015_0768.py``` and deploys a particle filter on a dataset of 

# PAR-FIT

Notice: These data were produced by Battelle Energy Alliance, LLC under Contract No. DE-AC07-05ID14517 with the Department of Energy. During the period of commercialization or such other time period specified by the Department of Energy, the Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this data to reproduce, prepare derivative works, and perform publicly and display publicly, by or on behalf of the Government. Subsequent to that period the Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this data to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. The specific term of the license can be identified by inquiry made to the Contractor or DOE. NEITHER THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
