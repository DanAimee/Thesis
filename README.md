# Thesis Project: Enzyme Kinetics with Machine Learning

Welcome to the GitHub repository for my thesis project! This project explores enzyme kinetics and parameter estimation using machine learning techniques, specifically applying NDEs and HMC Bayesian inference to enzymatic pathways. Below is an overview of the project, methodology, and structure of this repository.

Overview

Cleaning and scaling time-series NMR data.

Building and training Neural ODE models using the JAX-Catalax framework.

Conducting Bayesian inference via Hamiltonian Monte Carlo (HMC) for parameter estimation.

Validating results using cross-validation and Monte Carlo simulations.
Repository Structure
The repository is organized as follows:

Data/: Contains the raw and processed NMR time-series data used for model training and validation.

HMC/: Includes scripts and notebooks related to Hamiltonian Monte Carlo implementations for parameter estimation.

Optimization/: Houses optimization routines and algorithms applied during model training.

Pickles/: Stores serialized Python objects (e.g., trained models) for easy loading and reuse.

Time_Scaling/: Contains functions and scripts for time-scaling transformations applied to the data.

Training_Neural_ODE/: Includes notebooks and scripts for building and training the Neural ODE models.

Validation/: Holds validation scripts and results, including cross-validation and Monte Carlo simulation outputs.

Visualization/: Contains tools and scripts for visualizing data, model outputs, and results.


Visualization_Graphing.ipynb: Jupyter Notebook for generating visual representations of the data and model results.
