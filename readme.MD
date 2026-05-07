# Super-Level-Set Regression: Conditional Quantiles via Volume Minimization

This repository provides tools for constructing and evaluating conformal prediction sets for multivariate regression, implementing our proposed **Super-level-set regression** method. It accompanies our paper:

**Super-level-set regression: Conditional quantiles via volume minimization**.

## Installation

Clone this repository and install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Overview

The package is organized as follows:

### **code/**  
Contains the core implementation of SLS methods:
- `models.py`: Implements the `SLS_regressor` class for learning and evaluating highest density regions.
- `example_of_usage.ipynb`: Jupyter notebook demonstrating how to use the implemented methods.

### **experiments/**  
Contains all resources related to experimental evaluations:
- **code/**: Scripts to reproduce the experiments described in the paper.  
  To run an experiment, open a terminal in this folder and execute:
  ```bash
  python _generate_experiment.py <parameter_folder_name> <parameter_seed_number> 
  ```
  Example:
  ```bash
  python generate_experiment_projection.py taxi 1
  ```

- **parameters/**: JSON files specifying hyperparameters for various strategies.
- **figs/**: Plots and figures used in the paper.
- **results/**: Saved outputs from the experiments.

To generate a figure used in the paper, run the corresponding `05_to_figs.ipynb` or `06_comparison_quantile_minimization.ipynb` notebooks.


## Using `SLS_regressor`

### Basic usage for existing pointwise predictor


```python
tau = 0.9 # Desired coverage level

tau_param = TauParameterAnnealer(
                 tau,                
                 warm_start_step=500, 
                 tau_low_target_step=100, 
                 tau_low_steepness=1e-3,
                 tau_high_target_step=100, 
                 tau_high_steepness=1e-2,
                 low_error_init=0.5,   
                 low_error_max=0.01,    
                 high_error_init=0.2,  
                 high_error_max=0.01,  
                 eps=1e-5              
                 )

model = SLS_regressor(
                    dim_X=dim_X, 
                    dim_y=dim_y, 
                    cov_mode="full_cholesky", # or low_rank
                    num_flow_layers=0, # number of layers for the flows
                    K=1, # number of components
                    mode="full_cholesky" # or low_rank
                    )

model.fit(X_train, Y_train, 
          tau=tau, 
          epochs=num_epochs, 
          lr=lr, 
          batch_size=batch_size,  
          return_best=True,  # Returns the model with the lowest average marginal volume
          print_every=10,    # Frequency to print the loss
          tau_parameterAnnealer=tau_param, # The parameter annealing scheme
          loss_function="log_volume" # or full volume, or only_quantile or NLL
          )
```
This allows to learn the level sets for the scores introduced in the paper. 
It is also possible to calibrate the models and evaluate the volume (or induced proxy for the union of flows) by running

```python
model.conformalize(X_calibration, Y_calibration, tau)
vol = model.compute_average_volume(X_train[[0]])
```

The sets can be queried by running 
```python
mu, precision_params_list, partition, q = model.call_conformalize(x_tensor) # If you want to                                                                            #use the calculated quantile from conformal prediction
mu, precision_params_list, partition, q = model(x_tensor) # If you want to use the q(.) 
                                                          #learned during training

# mu -> the centers mu_k
# precision_params_list -> the matrices L_k either ('low_rank', D_norm, V_norm) or ('full_cholesky', L_norm)
# partition -> the weights for all partitions 
# q -> the estimated conditional quantiles
```



## Citation
If you use this repository for research purposes, please cite our paper:

**Super-Level-Set Regression: Conditional Quantiles via Volume Minimization**.

For any questions, feel free to contact us.