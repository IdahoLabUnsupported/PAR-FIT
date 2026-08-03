Resampling Functions
====================
This library contains the resampling functions used to repopulate the particle filter. There are three different resampling functions that can be used. 

Multinomial Resampler
----------------------------
This resampler draws N independent random numbers from a uniform distribution relative to the weights of each particle. 

.. math::
	p_i = \frac{w_i}{\sum_{j=0}^n w_j}

where :math:`p_i` is the probability of re-selecting a particle, :math:`i` to be resampled into the particle population.

Systematic Resampler
----------------------------
This method resamples particles based on their weights while maintaining diversity in particle values, preventing degeneracy. 

1. Generate evenly spaced positions with random offset between the interval :math:`[0,1]`, spaced by :math:`1/N`. The offset ensures randomness while keeping positions evenly distributed. 

2. For each position, find the particle index where the cumulative weight is greater than or equal to the spaced positions.

.. math::
	u_i = u_0 + \frac{i}{N}, i=0,...,N-1


Stratified Resampler
----------------------------
This method resamples particles from N equal partitions, with one random sample per partition. Each partition is drawn as follows:

.. math::
	u_i ~ U(\frac{i}{N},\frac{i+1}{N})

Threshold Resampler 
----------------------------
This method resamples particles randomly if a particle's weight is less than a user defined threshold. The default threshold is :math:`1/\text{number of particles}`. 

.. autofunction:: Resample.resample_multinomial
.. autofunction:: Resample.resample_systematic
.. autofunction:: Resample.resample_stratified
.. autofunction:: Resample.resample_threshold

Preprocessing Resampling Functions
============================
A preprocessing resampling function determines if resampling needs to occur. 

Residual Resampler
----------------------------
Not implemented yet. 

See https://stonesoup.readthedocs.io/en/v1.4/auto_tutorials/sampling/ResamplingTutorial.html for information on residual resampling.

Effective Sample Size (ESS) Resampler
----------------------------
The ESSResampler is a wrapper for other resampler functions. ESS checks at each time step whether particle resampling is needed. Resampling is only performed at a given time step if the Kish’s ESS criterion is used defined as:

.. math:: 
	ESS := (\sum_(i=1)^n w_i^2)^{-1} < \frac{N}{2} 
	
By default, a resample is performed if the threshold N⁄2 is exceeded. Here w_i is the normalized particle weight and N is the number of particles in the filter.   

.. autofunction:: Resample.resample_RR
.. autofunction:: Resample.resample_ESS
