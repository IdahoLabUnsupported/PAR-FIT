.. Particle Filter for Inference-based Testing (PAR-FIT) documentation master file, created by
   sphinx-quickstart on Wed Mar  4 09:04:27 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Particle Filter for Inference-based Testing (PAR-FIT)
=======================================================================

**PAR-FIT** is a Python Library used to determine the reliability of individual data-driven machine learning (ML) model predictions. It is intended to resolve trustworthiness issues in how ML models derive predictions from conventionally opaque decision functions. The software is ideal for ML integrated with real time systems (e.g., instrumentation systems) which require short latency between sensor measurement, model prediction, and evaluation of prediction reliability. Individual prediction reliability is evaluated by examining proximity and locality of underlying training data used to derive the prediction. The primary assumption is that nearby training data serves as inductive evidence to justify a model’s prediction achieves a desired functionality. The Sequential Probability Ratio Test is also used to generate a qualitative conclusion on reliability and is based on hypothesis testing.  

.. note:: 
	This project is under active development. 
	
.. toctree::
   :maxdepth: 2
   :caption: Contents 

   Introduction
   Installation
   Instructions
   Example
   Transient_Example
   
.. toctree::
   :maxdepth: 2
   :caption: Code Details  
   
   Swarm
   Particle
   Scoring
   Weighting
   Resample

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
