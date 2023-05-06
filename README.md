# Project

This study aims at constructing multi-class disease classification model using microbiome dataa, specifically focusing on liver cirrhosis, type II diabetes,
obesity, and irritable bowel syndrome.

## Data

* abundance_T.csv: Raw data (transposed)
* data_species.pkl/.csv: Cleaned data processed by data_process.ipynb
* test_set.pkl: testing data (30%)
* train_set.pkl: training data (70%)



## Notebooks

### data_process.ipynb

The ipynb file is the well documented process for data wrangling. It contains number encoding, single imputation, and sample filtering based on outcome classes.


### TabPFN.ipynb.ipynb

The ipynb file is the well documented process for TabPFN model & XGBoost model hyperparameter tuning & evaluation

### Tuning_visualization.ipynb

Visuzliation of the hyperparameter tuning process in FTTransformer using RayTune & NYU HPC system

### Model_evaluations.ipynb

This ipynb file evaluates the model performance of default & optimized FTTransformer models.

## Helper programs

### data_cleaner.py

A helper class to do data cleaning.

### DL_functions.py

A helper function sets to do model testing, evaluation, and visualization for FTTransformer

## Scripts

### FTT_optimized_runner.py

The script for 3-fold stratified CV FTTransformer model training

### FTT_ray_tune.py

The script for 200 random trials hyperparamter tuning for FTTransformer

### run_FTT_raytune.sh

The slurm job file for HPC system. Utilize singularity container for environment setup.
