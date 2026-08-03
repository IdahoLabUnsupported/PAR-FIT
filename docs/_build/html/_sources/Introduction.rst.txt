Introduction
============

Particle Filter for Inference Based Testing (PAR-FIT) is a generic method that uses particles filters on a training dataset to determine if a model (whether built with neural networks or others) is represented by the dataset. The particle filter can be used for several different purposes including uncertainty quantification of model predictions, anomaly detection for time-series predictions, forecasting, individual sensor anomaly detection, and out-of-distribution detection in real-time environments where speed and efficiency are required.

The particle filter relies on two assumptions:

1. Training data and/or operational data represent the baseline condition of the system. This data may have been used to train the model or collected during operation. Deviations from this baseline, whether occuring in a single sensor measurement or multiple, represent undesirable trends in the system that need to be detected. These deviations may stem from a variety of root causes, including but not limited to, miscalibration of sensors, sensor degradation, or human related errors.

2. The reliability of model prediction is relative to proximity to the baseline reference dataset or a baseline model (e.g., physics equation)


Deployment
----------

PAR-FIT is a standalone evaluation model that does not need to be developed in tandem with predictive models. If the training/operational dataset is known, and the model input and output are known, then PAR-FIT can make an evaluation of model predictions. This can be used in two different manners:

1. During model development where PAR-FIT may be used to gauge what data is being used in a prediction and whether that models prediction is an interpolation or extrapolation.

2. During model deployment, where PAR-FIT may be used to evaluate whether sensor(s) are anomalous or normal and the data uncertainty behind that prediction. PAR-FIT may be implemented in real-time or for post-processing purposes.


Beginning
---------

Start by navigating to the `Installation` tab to install the package. Various code examples can be found under `Example` and `Transient Example`.