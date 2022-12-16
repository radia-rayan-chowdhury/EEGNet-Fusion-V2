# Enhanced Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification

## Dependencies Required
-------------------------
* Python 3.7
* Tensorflow 2.1.0
* Pywavelets 1.1.1
* SciKit-learn 0.22.1
* Gumpy (https://github.com/gumpy-bci/gumpy)
* SciPy 1.4.1
* Numpy 1.18.1
* mlxtend 0.17.2
* statsmodels 0.11.1
* pyEDFlib 0.1.17

## Running
--------------
The model has evaluated using three publicly available dataset: 
The PhysioNet EEG Motor Movement/Imagery Dataset, the BCI Competition IV-2a and the BCI Competition IV-2b dataset.


A. Using the PhysioNet EEG Motor Movement/Imagery Dataset:
------------------------------------------------------------
The program can be run from the CLI with the following required arguments:

1) The number of subjects to be used from the dataset (integer)
2) The number of epochs the training of models should be done (integer)
3) The number of target classes in the classification (integer)
4) What type of trials should be extracted from the data (1 or 2, where 1 => executed trials only and 2 => imagined trials only)
5) If CPU-only mode should be used (True / False). Note that for GPU mode you will need to have CUDA installed.

Example: python run_experiments.py 109 100 2 1 True

NB: The EEG data has to be unpacked into the working directory "data" folder.

B. Using the BCI Competition IV-2a Dataset:
--------------------------------------------
1) Run the run_experiments_BCI_2a.py file

Example: python run_experiments_BCI_2a.py

NB: The dataset has to be unpacked into the working directory "BCI Competition IV" folder.

C. Using the BCI Competition IV-2b Dataset:
--------------------------------------------
1) Run the run_experiments_BCI_2a.py file

Example: python run_experiments_BCI_2b.py

NB: The dataset has to be unpacked into the working directory "BCI Competition IV 2b" folder.


The dataset Source:
-----------------------

	a. EEG Motor Movement/Imagery Dataset: https://physionet.org/content/eegmmidb/1.0.0/
	b. BCI Competition IV 2a Dataset: https://www.bbci.de/competition/iv/   or, http://bnci-horizon-2020.eu/database/data-sets
	c. BCI Competition IV 2b Dataset: https://www.bbci.de/competition/iv/   or, http://bnci-horizon-2020.eu/database/data-sets


---------------------------
Radia Rayan Chowdhury

