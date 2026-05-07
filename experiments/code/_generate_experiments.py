import json
import pickle
import argparse


import numpy as np
import matplotlib.pyplot as plt

from utils import *
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from data_loading import *

import json

from covmetrics import ERT, WSC
from tabicl import TabICLRegressor


from approaches.gaussian_predictor_likelihood import *
from approaches.gaussian_predictor_levelsets import *
from approaches.gaussian_trainer import *
from approaches.OT_predictor import *
from approaches.MVCS import *
from approaches.covariances import *
import pandas as pd
from pathlib import Path

from models import *

import sys
import os

# --- BLOC DE CONFIGURATION DU PATH ---

current_script_dir = os.path.dirname(os.path.abspath(__file__))
path_to_add = os.path.join(current_script_dir, "approaches", "multi")
if not os.path.exists(path_to_add):
    print(f"ERREUR CRITIQUE : Le dossier n'existe pas : {path_to_add}")
    print(f"Vérifie l'arborescence. Dossier actuel : {os.getcwd()}")
else:
    print(f"Succès : Ajout de {path_to_add} au Python Path")
    sys.path.insert(0, path_to_add)
# -------------------------------------

try:
    from moc.models.trainers.lightning_trainer import get_lightning_trainer
except ImportError as e:
    print("\n--- DIAGNOSTIC D'ERREUR ---")
    print(f"L'import a échoué : {e}")
    print("Vérifie que :")
    print("1. Le dossier 'approaches/multi/moc' existe.")
    print("2. Le dossier 'approaches/multi/moc' contient un fichier '__init__.py'.")
    print("3. Le dossier 'approaches/multi/moc/models' contient un fichier '__init__.py'.")
    sys.exit(1)


from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import C_HDR, L_CP, PCP, CP2_PCP_Linear, C_PCP

import time

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("config_name", type=str, help="Nom de la configuration")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    config_name = args.config_name
    experiment_index = args.experiment_index  

    seed_everything(experiment_index)

    print('config_name:', config_name)

    config_path = "../parameters/" + config_name + ".json"
    with open(config_path, 'r') as file : 
        parameters = json.load(file)

    tab_alpha = [0.1, 0.5]

    
    print(f"Experiment {experiment_index}/10")
    
    load_path = "../../data/processed_data/" + parameters["load_name"] + ".npz"
    X, Y = load_data(load_path)

    splits = [parameters["prop_train"], parameters["prop_stop"], parameters["prop_calibration"], parameters["prop_test"]]

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64

    subsets = split_and_preprocess(X, Y, splits=splits, normalize=True)

    x_train, y_train, x_calibration, y_calibration, x_test, y_test, x_stop, y_stop = subsets["X_train"], subsets["Y_train"], subsets["X_calibration"], subsets["Y_calibration"], subsets["X_test"], subsets["Y_test"], subsets["X_stop"], subsets["Y_stop"]

    print("X_train shape:", x_train.shape, "Y_train shape:", y_train.shape)
    print("X_cal shape:", x_calibration.shape, "Y_cal shape:", y_calibration.shape)
    print("X_test shape:", x_test.shape, "Y_test shape:", y_test.shape)
    print("X_stop shape:", x_stop.shape, "Y_stop shape:", y_stop.shape)

    
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]


    num_epochs = parameters["num_epochs"]
    batch_size = parameters["batch_size"]
    lr = parameters["lr"]

    dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64

    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    x_stop_tensor = torch.tensor(x_stop, dtype=dtype)
    y_stop_tensor = torch.tensor(y_stop, dtype=dtype)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=dtype)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=dtype)
    x_test_tensor = torch.tensor(x_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)
    x_train_and_stop_tensor = torch.concat([x_train_tensor, x_stop_tensor])
    y_train_and_stop_tensor = torch.concat([y_train_tensor, y_stop_tensor])
    
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size, shuffle=True)

    train_and_stop_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_and_stop_tensor, y_train_and_stop_tensor), batch_size= batch_size, shuffle=True)
    calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size= batch_size, shuffle=True)


    for alpha in tab_alpha:
        tau = 1-alpha

        tau_param_init = TauParameterAnnealer(tau,                
                        warm_start_step=float('inf')
                        )

        tau_param_fine_tune = TauParameterAnnealer(tau,                
                        warm_start_step=1, 
                        tau_low_target_step=100, 
                        tau_low_steepness=1e-3,
                        tau_high_target_step=100, 
                        tau_high_steepness=1e-2,
                        low_error_init=0.5,   
                        low_error_max=0.03,    
                        high_error_init=0.2,  
                        high_error_max=0.03,  
                        eps=1e-5              
                        )

        if output_dim < 5:
            cov_mode = "full_cholesky"
            loss_function = "full_volume"
        else:
            cov_mode = "low_rank"
            loss_function = "log_volume"
        model = UnifiedConditionalEstimator(dim_X=input_dim, dim_y=output_dim, 
                                            cov_mode=cov_mode, num_flow_layers=3, K=1,
                                            det_normalized=False
                                            )

        model.fit(x_train_tensor, y_train_tensor, 
                X_val = x_stop_tensor,
                y_val = y_stop_tensor,
                tau=tau, 
                epochs=num_epochs, 
                lr=lr, 
                batch_size=batch_size, 
                return_best=True, 
                print_every=100,
                tau_parameterAnnealer=tau_param_init,
                loss_function=loss_function
                )
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        model.fit(x_train_tensor, y_train_tensor, 
                X_val = x_stop_tensor,
                y_val = y_stop_tensor,
                tau=tau, 
                epochs=num_epochs, 
                lr=lr, 
                batch_size=batch_size, 
                return_best=True, 
                print_every=100,
                tau_parameterAnnealer=tau_param_fine_tune,
                loss_function=loss_function
                )
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # if input_dim < 100:
        #     print("Fitting a post-hoc TabICL quantile.")
        if input_dim < 100 or len(x_train_and_stop_tensor)>10_000:
            model.fit_external_quantile(TabICLRegressor(), x_train_and_stop_tensor, y_train_and_stop_tensor)
        else:
            model.fit_external_quantile(TabICLRegressor(batch_size=1), x_train_and_stop_tensor, y_train_and_stop_tensor)

        model.conformalize(x_calibration_tensor, y_calibration_tensor, tau)
        average_volume = model.compute_average_volume(x_test_tensor, scaled=True)
        print("Average volume level set:", average_volume)

        start = time.time()
        cover_level_sets = model.get_cover(x_test_tensor, y_test_tensor)
        end = time.time()
        time_method = end - start
        ERT_method = ERT().evaluate(x_test_tensor, cover_level_sets, alpha)

        coverage = cover_level_sets.mean().item()

        name = "level_sets"
        file_path = Path("../results/results_gpu_V5.csv")
        new_row = {
            "dataset": config_name,
            "experiment": experiment_index,
            "alpha": alpha,
            f"volume_{name}": np.array(average_volume),
            f"coverage_{name}": coverage,
            f"ERT_{name}": ERT_method,
            f"time_{name}": time_method,
        }

        if file_path.exists():
            df = pd.read_csv(file_path)
            
            mask = (
                (df["dataset"] == config_name) & 
                (df["experiment"] == experiment_index) & 
                (df["alpha"] == alpha)
            )
            
            if mask.any():
                for key, value in new_row.items():
                    df.loc[mask, key] = value
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        df.to_csv(file_path, index=False)
   

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # ignore tous les warnings dans ce bloc

        model = MQF2LightningModule(input_dim, output_dim)
        trainer = get_lightning_trainer( max_epochs=parameters.get("max_epochs_multi", 100) )
        trainer.fit(model, train_dataloaders=train_and_stop_loader)

        conformalizers = {
            'C-HDR': C_HDR,
            'PCP': PCP,
            'C-PCP': C_PCP,
            'CP2-PCP-Linear': CP2_PCP_Linear,
            'L-CP': L_CP,
        }

        # boucle automatique sur chaque conformalizer
        for name, conformalizer_class in conformalizers.items():
            print("Method :", name)
            conformalizer = conformalizer_class(calibrationloader, model)
            for alpha in tab_alpha:  
                start = time.time()
                cover = compute_coverage_indicator(conformalizer, alpha, x_test_tensor, y_test_tensor)
                print("cover ok")
                end = time.time()
                all_volumes = compute_log_region_size(conformalizer, model, alpha, x_test_tensor)
                volumes = torch.mean(torch.exp(all_volumes)**(1/output_dim)).item()
                print("volume", volumes)
                coverage = torch.mean(cover).item()

                print('coverage', coverage)
                time_method = end - start

                ERT_method = ERT().evaluate(x_test_tensor, cover, alpha)
                
                file_path = Path("../results/results_gpu_V5.csv")
                new_row = {
                    "dataset": config_name,
                    "experiment": experiment_index,
                    "alpha": alpha,
                    f"volume_{name}": np.array(volumes),
                    f"coverage_{name}": coverage,
                    f"ERT_{name}": ERT_method,
                    f"time_{name}": time_method,
                }

                if file_path.exists():
                    df = pd.read_csv(file_path)
                    
                    mask = (
                        (df["dataset"] == config_name) & 
                        (df["experiment"] == experiment_index) & 
                        (df["alpha"] == alpha)
                    )
                    
                    if mask.any():
                        for key, value in new_row.items():
                            df.loc[mask, key] = value
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    df = pd.DataFrame([new_row])
                df.to_csv(file_path, index=False)

            print("\n\n##########################################\n\n")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()