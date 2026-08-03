Installation
============
It is recommend to have a environment manager when utilizing PAR-FIT. Anaconda is the recommended; the link to Anaconda can be found here:
https://www.anaconda.com/download

Base Installation through Environment File
------------------------------------------
A yaml environment file is provided for quick installation and test purposes. This yaml environment provides only the code packages needed to run the algorithm and does not include debugging or visualization tools. In an conda command line run the following lines. 

.. code-block:: python

	conda env create -f environment.yml
	
Pip Installation through Requirements File
------------------------------------------
A requirements.txt file is provided for an alternative method of installation the necessary packages.

.. code-block:: python

	pip install -r requirements.txt

Installation for Comparative Methods
------------------------------------
This repository also provides two other comparative methods to evaluate the efficacy of the particle filter. They include the Multivariate State Estimation Technique (MSET) and the Laplacian Exponential Decay Kernel for proximity detection. To run these examples, follow the installation instructions below.

Matplotlib.pyplot is used for the visualization of functions within the code. Installation can be completed within conda or in a python environment.

.. code-block:: python

	conda install matplotlib
	
While any IDE may be used to run the code, Spyder is the default IDE that the code was developed on. Spyder can be installed natively with conda or with the command line below. Newer versions of Spyder can be used.

.. code-block:: python

	conda install anaconda::spyder==5.5.1
	
If you are developing or testing the provided neural network models, then pytorch will be needed. The below command line specifies installation with CPU only. Other configurations are available at: https://pytorch.org/get-started/locally/

.. code-block:: python

	pip3 install torch torchvision
	


