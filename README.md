# softcoding_ECOC
Two types of Soft-Coding methods to improve ECOC 
This is the implementation for paper: [The Design of Soft Recoding Based Strategies for Improving Error-Correcting Output Codes]

## Acknowledgement
* The classifier and the evaluation function is modified from [scikit-learn 0.22](https://scikit-learn.org/stable/)

## Environment
* **Windows 10 64 bit**
* **Python 3**
* [Anaconda](https://www.anaconda.com/) is strongly recommended, all necessary python packages for this project are included in the Anaconda.

## Function
The main implementation function of soft coding is in the  `soft_matrix.py`.
* _soften_matrix: This is the ECOC coding generation function of MVR in the soft-coding
mode. Each code in the generated ECOC matrix is a real number with decimals after softening.
* _interval_matrix: This is the ECOC coding generation function of IR in the soft-coding
mode. Each code in the generated ECOC matrix becomes an interval, and the interval boundary
is a real number with decimals.
* _vector_score_soft: calculate the minimum distance of the output vector to each class (using soft value method)
* _vector_score_interval: calculate the score of the output vector to each class (using soft value interval method)

There are several ready-made ECOC encoding methods for softening listed in `Classifier.py`.
Runner can call different ECOC codes through `Classifier.py`.
* OVA_ECOC
* OVO_ECOC
* Sparse_random_ECOC
* Dense_random_ECOC
* D_ECOC
* AGG_ECOC

## Dataset
* Data format  
Data sets of uci and microarray are both included into the folder `($root_path)/DataSet/test_set`.
The dataset included in the folder must have no null/nan/? values.
The datasets will be loaded into algorithm in `test_ECOC_package.py`.
* Data processing  
The data will automatically undergo simple data preprocessing.
The data will use `sklearn.Imputer` to fill in the missing values in the mean.
Standardize all features through `sklearn.preprocessing.StandardScaler`.

## Runner Setup
The `test_ECOC_package.py` calls the softcoding_ECOC. 
  
The are some variables to control algorithm.
* `REPEAT_TIMES`: Run Times(The results will be average of run times of results)
* `base_leaner`: base classifier of algorithm
* `clf_soft_interval`: soft coding ECOC classifier of IR, its base ECOC coding type can be called by `Classifier` class.
And the coding types of **hard**, **MVR**, **IR** by setting classifier's parameters *soften* and *soft_value*.
* `clf_soft_value`: soft coding ECOC classifier of MVR, its base ECOC coding type can be called by `Classifier` class.
And the coding types of **hard**, **MVR**, **IR** by setting classifier's parameters *soften* and *soft_value*.
* `clf_hard`: hard coding ECOC classifier

The results are collected and written in the filefolder `softcoding_report`.