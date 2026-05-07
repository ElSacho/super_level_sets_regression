import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
from torch.utils.data import DataLoader, TensorDataset

        
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
from torch.utils.data import DataLoader, TensorDataset

import torch

def compute_adaptive_p(q, S, lower_target_ratio, upper_target_ratio):
    diff = q.detach() - S.detach()
    sq_diff = diff**2
    p = torch.zeros_like(diff)
    
    def compute_side(mask, target_ratio):
        num_valid = mask.sum()
        
        if num_valid == 0:
            return
        if num_valid == 1:
            p[mask] = 1.0
            return
        
        flat_mask = mask.flatten()
        flat_sq_diff = sq_diff.flatten()
        masked_sq = torch.where(flat_mask, flat_sq_diff, torch.tensor(float('inf'), device=q.device))
        
        # --- NOUVEAU : Gestion du cas où le ratio est 0 (ou presque) ---
        # On utilise 1e-7 comme marge de sécurité pour les erreurs d'arrondi des flottants
        if target_ratio <= 1e-7:
            # On trouve la plus petite distance
            min_val = torch.min(masked_sq)
            # Seuls les éléments correspondant à cette distance minimale reçoivent 1.0
            is_min = (sq_diff == min_val) & mask
            p[is_min] = 1.0
            return
        # ---------------------------------------------------------------
        
        # Le reste de la logique normale pour un ratio > 0
        vals, _ = torch.topk(masked_sq, k=2, largest=False)
        d1_sq, d2_sq = vals[0], vals[1]
        
        if d1_sq == d2_sq:
            coef = torch.tensor(1.0, device=q.device)
        else:
            coef = torch.log(torch.as_tensor(target_ratio, device=q.device)) / (d1_sq - d2_sq)
            
        p[mask] = torch.exp(-coef * (sq_diff[mask] - d1_sq))

    # Application asymétrique
    compute_side(diff >= 0, lower_target_ratio)
    compute_side(diff < 0, upper_target_ratio)
    
    return p


class TauParameterAnnealer:
    def __init__(self, 
                 tau,                  # Fixed central tau
                 warm_start_step=3, 
                 tau_low_target_step=10, 
                 tau_low_steepness=1e-3,
                 tau_high_target_step=100, 
                 tau_high_steepness=1e-2,
                 low_error_init=0.5,   # Distance below tau at start
                 low_error_max=0.1,    # Distance below tau at the end
                 high_error_init=0.2,  # Distance above tau at start
                 high_error_max=0.02,   # Distance above tau at the end
                 eps=1e-5              # Safety margin
                 ):
        
        self.warm_start_step = warm_start_step
        self.current_step = 0
        self.eps = eps
        
        # Helper function to clamp values strictly between [eps, 1 - eps]
        self.clamp = lambda x: max(self.eps, min(1.0 - self.eps, x))
        
        # 1. Store the fixed, clamped tau
        self.tau = self.clamp(tau)
        
        # 2. Calculate absolute bounds based on tau and the error margins
        # Using subtraction for 'low' assuming you pass positive margin values
        self.tau_low_init = self.clamp(self.tau - low_error_init)
        self.tau_low_max = self.clamp(self.tau - low_error_max)
        
        self.tau_high_init = self.clamp(self.tau + high_error_init)
        self.tau_high_max = self.clamp(self.tau + high_error_max)
        
        # Paramètres pour tau_low
        self.tau_low_k = tau_low_steepness
        self.tau_low_t0 = tau_low_target_step
        self.tau_low_min = 1 / (1 + math.exp(self.tau_low_k * self.tau_low_t0))
        
        # Paramètres pour tau_high
        self.tau_high_k = tau_high_steepness
        self.tau_high_t0 = tau_high_target_step
        self.tau_high_min = 1 / (1 + math.exp(self.tau_high_k * self.tau_high_t0))
        
        # Initialisation aux valeurs de départ
        self.tau_low = self.tau_low_init
        self.tau_high = self.tau_high_init

    def step(self):
        self.current_step += 1
        
        if self.current_step > self.warm_start_step:
            t = self.current_step - self.warm_start_step
            
            # --- Calcul pour tau_low ---
            raw_tau_low = 1 / (1 + math.exp(-self.tau_low_k * (t - self.tau_low_t0))) 
            norm_tau_low = (raw_tau_low - self.tau_low_min) / (1.0 - self.tau_low_min)
            new_tau_low = self.tau_low_init + (self.tau_low_max - self.tau_low_init) * norm_tau_low
            
            # Apply safety clamp against eps boundaries
            self.tau_low = self.clamp(new_tau_low)
            
            # --- Calcul pour tau_high ---
            raw_tau_high = 1 / (1 + math.exp(-self.tau_high_k * (t - self.tau_high_t0))) 
            norm_tau_high = (raw_tau_high - self.tau_high_min) / (1.0 - self.tau_high_min)
            new_tau_high = self.tau_high_init + (self.tau_high_max - self.tau_high_init) * norm_tau_high
            
            # Apply safety clamp against eps boundaries
            self.tau_high = self.clamp(new_tau_high)
            
        return self.tau_low, self.tau_high

    def get_params(self):
        return self.tau_low, self.tau_high
  
class ConditionalAdditiveCouplingLayer(nn.Module):
    def __init__(self, d, dim_X, hidden_dim=64, even=True, dropout= 0.1):
        super().__init__()
        self.even = even
        self.split_size = d // 2

        y1_dim = self.split_size if self.even else d - self.split_size
        in_dim = y1_dim + dim_X 
        out_dim = d - self.split_size if self.even else self.split_size

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=dropout),        
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # def forward(self, y, X):
    #     if self.even:
    #         y1, y2 = y[:, :self.split_size], y[:, self.split_size:]
    #         net_in = torch.cat([y1, X], dim=1) 
    #         z2 = y2 + self.net(net_in)
    #         return torch.cat([y1, z2], dim=1)
    #     else:
    #         y1, y2 = y[:, self.split_size:], y[:, :self.split_size]
    #         net_in = torch.cat([y1, X], dim=1)
    #         z2 = y2 + self.net(net_in)
    #         return torch.cat([z2, y1], dim=1)
    def forward(self, y, X):
        split_size = self.split_size
        if self.even:
            y1, y2 = y[:, :split_size], y[:, split_size:]
        else:
            y1, y2 = y[:, split_size:], y[:, :split_size]

        # 1. Prédiction normale
        net_in = torch.cat([y1, X], dim=1)
        shift = self.net(net_in)

        # 2. Prédiction à l'origine (y1 = 0)
        y1_zero = torch.zeros_like(y1)
        net_in_zero = torch.cat([y1_zero, X], dim=1)
        shift_zero = self.net(net_in_zero)

        # 3. On soustrait la composante de translation pure
        z2 = y2 + (shift - shift_zero)

        if self.even:
            return torch.cat([y1, z2], dim=1)
        else:
            return torch.cat([z2, y1], dim=1)

class ConditionalVolumePreservingFlow(nn.Module):
    def __init__(self, d, dim_X, num_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAdditiveCouplingLayer(d, dim_X, even=(i % 2 == 0), dropout=dropout) 
            for i in range(num_layers)
        ])

    def forward(self, y, X):
        z = y
        for layer in self.layers:
            z = layer(z, X) 
        return z

class RobustPrecisionHead(nn.Module):
    def __init__(self, input_dim, y_dim, init_sigma=1.0, mode="full_cholesky"):
        super().__init__()
        self.y_dim = y_dim
        self.mode = mode
        
        if mode == 'low_rank':
            self.rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * self.rank)
            
        elif mode == "full_cholesky":
            if y_dim > 10: 
                print("Warning: Large output dimension, mode = 'low_rank' is recommended.")
            num_chol = (y_dim * (y_dim + 1)) // 2
            self.fc_chol = nn.Linear(input_dim, num_chol)
            self.register_buffer('tril_indices', torch.tril_indices(y_dim, y_dim))
            
            with torch.no_grad():
                diag_mask = (self.tril_indices[0] == self.tril_indices[1])
                inv_softplus = math.log(math.exp(init_sigma) - 1)
                self.fc_chol.bias[diag_mask] = inv_softplus
        else:
            raise ValueError("The mode must either be 'full_cholesky' or 'low_rank'.")

    def forward(self, x):
        B = x.shape[0]
        if self.mode == 'low_rank':
            D = torch.exp(self.fc_log_diag(x)) + 1e-6
            V = self.fc_factors(x).view(B, self.y_dim, self.rank)
            
            # --- Normalize determinant to 1 ---
            log_det_D = torch.sum(torch.log(D), dim=1, keepdim=True)
            D_inv_V = V / D.unsqueeze(-1) 
            V_T_D_inv_V = torch.bmm(V.transpose(1, 2), D_inv_V) 
            I = torch.eye(V.shape[-1], device=x.device).unsqueeze(0).expand(B, -1, -1)
            _, logdet_I_V = torch.linalg.slogdet(I + V_T_D_inv_V) 
            
            # log(det(Omega)) = log(det(D)) + log(det(I + V^T D^-1 V))
            log_det_Omega = log_det_D + logdet_I_V.unsqueeze(-1)
            
            # Scale factor c so that c^d * det(Omega) = 1
            c = torch.exp(-log_det_Omega / self.y_dim)
            
            # Apply scaling: Omega_new = c * Omega  =>  D_new = c*D, V_new = sqrt(c)*V
            D_norm = D * c
            V_norm = V * torch.sqrt(c)
            
            return ('low_rank', D_norm, V_norm)
            
        else:
            chol_flat = self.fc_chol(x)
            L = torch.zeros(B, self.y_dim, self.y_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = chol_flat
            diag_idx = torch.arange(self.y_dim, device=x.device)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
            
            # --- Normalize determinant to 1 ---
            log_diag = torch.log(L[:, diag_idx, diag_idx])
            mean_log_diag = torch.mean(log_diag, dim=1, keepdim=True)
            
            # Scale factor c_L so that prod(diag * c_L) = 1
            c_L = torch.exp(-mean_log_diag).unsqueeze(-1)
            L_norm = L * c_L
            
            return ('full_cholesky', L_norm)
       
class RobustPrecisionHeadWithDet(nn.Module):
    def __init__(self, input_dim, y_dim, init_sigma=1.0, mode="full_cholesky"):
        super().__init__()
        self.y_dim = y_dim
        self.mode = mode
        
        if mode == 'low_rank':
            self.rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * self.rank)
            
        elif mode == "full_cholesky":
            if y_dim > 10: 
                print("Warning: Large output dimension, mode = 'low_rank' is recommended.")
            num_chol = (y_dim * (y_dim + 1)) // 2
            self.fc_chol = nn.Linear(input_dim, num_chol)
            self.register_buffer('tril_indices', torch.tril_indices(y_dim, y_dim))
            
            with torch.no_grad():
                diag_mask = (self.tril_indices[0] == self.tril_indices[1])
                inv_softplus = math.log(math.exp(init_sigma) - 1)
                self.fc_chol.bias[diag_mask] = inv_softplus
        else:
            raise ValueError("The mode must either be 'full_cholesky' or 'low_rank'.")

    def forward(self, x):
        B = x.shape[0]
        if self.mode == 'low_rank':
            D = torch.exp(self.fc_log_diag(x)) + 1e-6
            V = self.fc_factors(x).view(B, self.y_dim, self.rank)
            return ('low_rank', D, V)
        else:
            chol_flat = self.fc_chol(x)
            L = torch.zeros(B, self.y_dim, self.y_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = chol_flat
            diag_idx = torch.arange(self.y_dim, device=x.device)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
            return ('full_cholesky', L)

class MultipleQuantiles(nn.Module):
    def __init__(self, dim_X, hidden_dim=64, n_hidden_layer_quantile=3, dropout_quantile=0.3, eps=1e-6):
        super().__init__()
        layers = []
        layers.extend([
            nn.Linear(dim_X, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_quantile)
        ])
        for _ in range(n_hidden_layer_quantile - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_quantile)
            ])
        self.share_net = nn.Sequential(*layers)
        self.log_quantile_low = nn.Linear(hidden_dim, 1)
        self.log_quantile_tau = nn.Linear(hidden_dim, 1)
        self.log_quantile_high = nn.Linear(hidden_dim, 1)
        self.eps = eps

    def reset_parameters(self):
        """Re-initializes the weights of all layers in the network."""
        def _reset_weights(m):
            # Check if the module has the method AND ensure it's not the parent module itself
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
        # Apply the reset function to self and all children modules
        self.apply(_reset_weights)
    
    def forward(self, x):
        share_values = self.share_net(x)
        log_quantile_low  = self.log_quantile_low(share_values)
        log_quantile_tau  = self.log_quantile_tau(share_values)
        log_quantile_high = self.log_quantile_high(share_values)
        return torch.exp(log_quantile_low) + self.eps, torch.exp(log_quantile_tau) + self.eps, torch.exp(log_quantile_high) + self.eps
    
    def forward_tau(self, x):
        share_values = self.share_net(x)
        log_quantile_tau  = self.log_quantile_tau(share_values)
        return torch.exp(log_quantile_tau) + self.eps
    
    def forward_bound(self, x):
        share_values = self.share_net(x)
        log_quantile_low  = self.log_quantile_low(share_values)
        log_quantile_high = self.log_quantile_high(share_values)

        return torch.exp(log_quantile_low) + self.eps, torch.exp(log_quantile_high) + self.eps

class UnifiedConditionalEstimator(nn.Module):
    """
    Classe maîtresse unifiant K Flux (Flows) avec agrégation Softmin pour 
    modéliser une union de régions de confiance avec matrices de précision libres.
    """
    def __init__(self, dim_X, dim_y, K=3, det_normalized=True, use_partition=True, hidden_dim=64, n_hidden_layer=2, n_hidden_layer_quantile=1, num_flow_layers=1, cov_mode="full_cholesky", dropout=0.1, dropout_quantile=0.3):
        super().__init__()
        self.d = dim_y
        self.K = K       
        self.det_normalized = det_normalized
        self.external_quantile = False # Whether to use our quantile function or another model to predict the quantiles
        
        ### Create the share network for center, matrix and flow
        layers = []
        layers.extend([
            nn.Linear(dim_X, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])
        for _ in range(n_hidden_layer - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ])
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.shared_net = nn.Sequential(*layers)

        ### All the centers 
        self.head_mu = nn.Linear(hidden_dim, self.K * self.d)

        ### All the matrices
        if det_normalized:
            self.head_precisions = nn.ModuleList([
                RobustPrecisionHead(hidden_dim, self.d, mode=cov_mode)
                for _ in range(K)
            ])
        else:
            self.head_precisions = nn.ModuleList([
                RobustPrecisionHeadWithDet(hidden_dim, self.d, mode=cov_mode)
                for _ in range(K)
            ])
        
        ### Declare the volume preserving flows
        self.flows = nn.ModuleList([
            ConditionalVolumePreservingFlow(self.d, dim_X, num_layers=num_flow_layers, dropout=dropout) 
            for _ in range(K)
        ])

        ### Use the partition of the space only when asked and using multiple flows without determinant
        if K == 1 or not det_normalized: 
            use_partition = False
        self.use_partition = use_partition
        if use_partition:
            self.partition_score = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, K),
                nn.Softmax(dim=-1)
            )
        else:
            self.partition_score = None
        
        ### Quantile manager to predict the quantiles
        self.quantile_manager = MultipleQuantiles(dim_X=dim_X, 
                                             hidden_dim=hidden_dim, 
                                             n_hidden_layer_quantile=n_hidden_layer_quantile, 
                                             dropout_quantile=dropout_quantile
                                             )
        self.q_alpha = None

    def forward(self, X):
        h = self.shared_net(X)
        B = X.shape[0]
        
        mu_flat = self.head_mu(h)
        mu = mu_flat.view(B, self.K, self.d)
    
        if self.external_quantile:
            q = self.external_quantile_model.predict(X, output_type="quantiles", alphas=[self.tau])
            q = torch.tensor(q, dtype=torch.float32)
            q = q.view(-1, 1)
        else:
            q = self.quantile_manager.forward_tau(X)

    
        precision_params_list = [head(h) for head in self.head_precisions]

        if self.K > 1 and self.det_normalized and self.use_partition:
            partition = self.partition_score(h)
            # raw_partition = self.partition_score(h)
            # eps = 1e-3
            # partition = raw_partition * (1 - eps) + (eps / self.K) 
        else:
            partition = None
        
        return mu, precision_params_list, partition, q
    
    def get_q_kernel(self, X):
        raw_q_low, raw_q_high = self.quantile_manager.forward_bound(X)
    
        q_low = torch.min(raw_q_low, raw_q_high)
        q_high = torch.max(raw_q_low, raw_q_high)

        return q_low, q_high
    
    def step_parameter(self):
        self.num_iteration += 1
        self.beta += self.lr_beta
        self.beta = min(self.beta_max, self.beta)
    
    def call_conformalize(self, X):
        if self.q_alpha is None:
            raise Exception("Call the conformalize first")
        mu, precision_params_list, partition, q = self(X)
        if self.conformal_score == "additive":
            return mu, precision_params_list, partition, q + self.q_alpha
        elif self.conformal_score == "multiplicative":
            return mu, precision_params_list, partition, q * self.q_alpha
            
    def get_cover(self, X, y):
        scores = self.get_normalized_scores(X, y)
        cover = scores <= self.q_alpha
        return cover*1.0
       
    def get_frontiers(self, X, y, use_conformalize=False):        
        if use_conformalize: mu, precision_params_list, partition, q = self.call_conformalize(X)
        else: mu, precision_params_list, partition, q = self(X)

        G_list = []
        log_det_L_list = []

        for k in range(self.K):
            z_k = self.flows[k](y, X)
            mu_k = mu[:, k, :]
            
            diff_k = (z_k - mu_k).unsqueeze(-1)
            
            if partition is not None:
                p_k = partition[:, k:k+1] + 1e-8 
            
            mode_k = precision_params_list[k][0]

            if mode_k == 'full_cholesky':
                L_k = precision_params_list[k][1]
                L_diff_k = torch.bmm(L_k, diff_k).squeeze(-1)
                G_k_raw = torch.sum(L_diff_k ** 2, dim=1)
                                
                idx = torch.arange(self.d, device=X.device)
                log_det_L_k_raw = torch.sum(torch.log(L_k[:, idx, idx]), dim=1)
                
            elif mode_k == 'low_rank':
                D_k, V_k = precision_params_list[k][1], precision_params_list[k][2]
                
                diff_sq_k = diff_k.squeeze(-1) ** 2
                G_k_raw = torch.sum(D_k * diff_sq_k, dim=1) + torch.sum(torch.bmm(V_k.transpose(1, 2), diff_k).squeeze(-1) ** 2, dim=1)
                
                log_det_D_k = torch.sum(torch.log(D_k + 1e-6), dim=1)
                D_inv_V_k = V_k / (D_k.unsqueeze(-1) + 1e-6) 
                V_T_D_inv_V_k = torch.bmm(V_k.transpose(1, 2), D_inv_V_k) 
                I = torch.eye(V_k.shape[-1], device=X.device).unsqueeze(0).expand(X.shape[0], -1, -1)
                _, logdet_k = torch.linalg.slogdet(I + V_T_D_inv_V_k) 
                log_det_L_k_raw = 0.5 * (log_det_D_k + logdet_k)

            if partition is not None and self.tau_parameterAnnealer.warm_start_step < self.num_iteration:
                d = y.shape[-1]
                p_k_squeeze = p_k.squeeze(-1)
                # G_k = G_k_raw - torch.log(p_k_squeeze)
                # log_det_L_k = log_det_L_k_raw - torch.log(p_k_squeeze)
                G_k = G_k_raw / (p_k_squeeze )**(2.0 / d)
                regularization = max((self.tau_parameterAnnealer.warm_start_step - self.num_iteration), 0) * torch.log(p_k_squeeze) 
                # log_det_L_k = log_det_L_k_raw - torch.log(p_k_squeeze) + regularization - regularization.detach()
                log_det_L_k = log_det_L_k_raw - torch.log(p_k_squeeze) 
            else:
                G_k = G_k_raw 
                log_det_L_k = log_det_L_k_raw 

            G_list.append(G_k)
            log_det_L_list.append(log_det_L_k)

        log_det_L_stacked = torch.stack(log_det_L_list, dim=1)
        G_stacked = torch.stack(G_list, dim=1)

        if self.K > 1:
            # G = - (1.0 / self.beta) * torch.logsumexp(-self.beta * G_stacked, dim=1, keepdim=True)
            # Calculate softmax weights (dim=1 is the component dimension K)
            weights = torch.softmax(-self.beta * G_stacked, dim=1)
            G = torch.sum(weights * G_stacked, dim=1, keepdim=True)
        else:
            G = G_stacked # TODO: check this

        return G, log_det_L_stacked, q
    
    def get_det_L(self, X, use_conformalize=False):        
        if use_conformalize: mu, precision_params_list, partition, q = self.call_conformalize(X)
        else: mu, precision_params_list, partition, q = self(X)

        log_det_L_list = []

        for k in range(self.K):
            
            if partition is not None:
                p_k = partition[:, k:k+1] + 1e-8 
            
            mode_k = precision_params_list[k][0]

            if mode_k == 'full_cholesky':
                L_k = precision_params_list[k][1]
                idx = torch.arange(self.d, device=X.device)
                log_det_L_k_raw = torch.sum(torch.log(L_k[:, idx, idx]), dim=1)
                
            elif mode_k == 'low_rank':
                D_k, V_k = precision_params_list[k][1], precision_params_list[k][2]
                                
                log_det_D_k = torch.sum(torch.log(D_k + 1e-6), dim=1)
                D_inv_V_k = V_k / (D_k.unsqueeze(-1) + 1e-6) 
                V_T_D_inv_V_k = torch.bmm(V_k.transpose(1, 2), D_inv_V_k) 
                I = torch.eye(V_k.shape[-1], device=X.device).unsqueeze(0).expand(X.shape[0], -1, -1)
                _, logdet_k = torch.linalg.slogdet(I + V_T_D_inv_V_k) 
                log_det_L_k_raw = 0.5 * (log_det_D_k + logdet_k)

            if partition is not None and self.tau_parameterAnnealer.warm_start_step < self.num_iteration:
                p_k_squeeze = p_k.squeeze(-1)
                log_det_L_k = log_det_L_k_raw - torch.log(p_k_squeeze)
            else:
                d = mu.shape[-1]
                log_det_L_k = log_det_L_k_raw 
            log_det_L_list.append(log_det_L_k)

        log_det_L_stacked = torch.stack(log_det_L_list, dim=1)

        return log_det_L_stacked, q

    def get_normalized_scores(self, X, y, batch_size=1000):
        frontiers_list = []
        q_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                frontiers_batch, _, q_batch = self.get_frontiers(X_batch, y_batch)
                frontiers_list.append(frontiers_batch.detach()) 
                q_list.append(q_batch.detach()) 
                
            frontiers = torch.cat(frontiers_list, dim=0)
            q = torch.cat(q_list, dim=0)
        
        
        # S, _, q = self.get_frontiers(X, y)
        return (frontiers / q).squeeze() 

    def compute_volume(self, X, batch_size = 1_000):
        """
        Calcule le volume exact approché (somme des composantes) pour un (ou plusieurs) point(s) X.
        """
        #TODO
        if self.q_alpha is None:
            raise Exception("Call the conformalize first")
        self.eval()

        det_list = []
        q_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                det_L_batch, q_batch = self.get_det_L(X_batch, use_conformalize=True)
                det_list.append(det_L_batch.detach()) 
                q_list.append(q_batch.detach()) 
                
            log_det_L_stacked = torch.cat(det_list, dim=0)
            q_conformalize = torch.cat(q_list, dim=0)

            # log_det_L_stacked, q_conformalize = self.get_det_L(X, use_conformalize=True)

            log_vol_penalty = torch.logsumexp(-log_det_L_stacked, dim=1, keepdim=True)
            log_V_constant = math.log((math.pi ** (self.d / 2)) / math.gamma(self.d / 2 + 1))
            log_vol_k = log_V_constant + (self.d / 2) * torch.log(q_conformalize) + log_vol_penalty
            total_vol = torch.exp(log_vol_k)
            
        return total_vol.squeeze()
    
    def compute_average_volume(self, X, scaled=False):
        volumes = self.compute_volume(X)
        return torch.mean(volumes) if not scaled else torch.mean(volumes**(1/self.d))

    def conformalize(self, X_cal, y_cal, alpha, conformal_score="multiplicative", fake_for_trial=False):
        self.conformal_score = conformal_score
        if fake_for_trial:
            print('NO CONFORMALIZATION AS ASKED')
            if conformal_score == "multiplicative":
                self.q_alpha = 1.0
            elif conformal_score == "additive":
                self.q_alpha = 0.0
            else:
                raise ValueError("The conformal score is not well defined.")
            return
        self.eval()
        with torch.no_grad():
            n = X_cal.shape[0]
            
            scores = self.get_normalized_scores(X_cal, y_cal) 
            p = int(np.ceil((n + 1) * (1 - alpha)))
            scores = torch.sort(scores, descending=True).values

            self.q_alpha = scores[p].item()  
                                        
            print(f"Conformalisation terminée (n={n}, alpha={alpha}). Multiplicateur q_alpha = {self.q_alpha:.4f}")

    
    def fit(
            self, 
            X_train, 
            y_train, 
            tau, 
            X_val=None, 
            y_val=None, 
            epochs=1000, 
            lr=1e-3, 
            batch_size=32, 
            weight_decay=1e-4, 
            beta = 5,
            beta_max = 100,
            lr_beta=1e-3,
            loss_function="log_volume",
            return_best=True, 
            print_every=1, 
            tau_parameterAnnealer=None
            ):        

        self.loss_function = loss_function
        self.tau = tau

        self.num_iteration = 0
        self.beta = beta
        self.beta_max = beta_max
        self.lr_beta = lr_beta

        if tau_parameterAnnealer is None:
            print("Using default fit param")
            self.tau_parameterAnnealer = TauParameterAnnealer(tau)  
        else:
            self.tau_parameterAnnealer = tau_parameterAnnealer
        
        # Dataloaders preparation
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

        # Keep track of the best model
        self.best_valid_vol = float('inf')
        self.best_model_weights = None

        for epoch in range(epochs):
            # Training phase
            self.train()
            total_train_loss = 0.0
            epoch_coverage = 0.0
            epoch_valid_volume = 0.0
            total_train_pb_loss = 0.0
            
            for batch_X, batch_y in train_loader:
            
                ### Treat the quantiles
                optimizer.zero_grad()
            
                loss_tau = self.compute_pb_loss(batch_X, batch_y, tau)
                loss_low, loss_high = self.compute_pb_kernel_loss(batch_X, batch_y)
                loss_pb = loss_low + loss_high + loss_tau
                loss_pb.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                ### Treat the shape
                optimizer.zero_grad()
                
                loss, batch_coverage, batch_valid_volume = self.compute_loss(batch_X, batch_y, tau)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                ### Updated the parameters (beta, phi(n), psi(n))
                self.step_parameter()
                                
                total_train_loss += loss.item() * batch_X.size(0)
                epoch_coverage += batch_coverage.item() * batch_X.size(0)
                epoch_valid_volume += batch_valid_volume.item() * batch_X.size(0)
                total_train_pb_loss += loss_pb.item() * batch_X.size(0)
                
            avg_train_loss = total_train_loss / len(X_train)
            avg_train_cov = epoch_coverage / len(X_train)
            avg_train_vol = epoch_valid_volume / len(X_train)
            avg_train_pb_loss = total_train_pb_loss / len(X_train)

            # Early stopping is performed on the volume - used if no validation set
            current_target_vol = avg_train_vol 

            # Validation if possible
            if has_val:
                self.eval()
                val_epoch_coverage = 0.0
                val_epoch_valid_volume = 0.0
                total_val_loss = 0.0
                total_val_pb_loss = 0.0
                
                with torch.no_grad():
                    for v_batch_X, v_batch_y in val_loader:
                        
                        loss_pb = self.compute_pb_loss(v_batch_X, v_batch_y, tau)
                        loss, v_batch_coverage, v_batch_valid_volume = self.compute_loss(v_batch_X, v_batch_y, tau)

                        val_epoch_coverage += v_batch_coverage.item() * v_batch_X.size(0)
                        val_epoch_valid_volume += v_batch_valid_volume.item() * v_batch_X.size(0)
                        total_val_loss += loss.item() * v_batch_X.size(0)
                        total_val_pb_loss += loss_pb.item() * v_batch_X.size(0)
                        
                avg_val_cov = val_epoch_coverage / len(X_val)
                avg_val_vol = val_epoch_valid_volume / len(X_val)
                avg_val_loss = total_val_loss / len(X_val)
                avg_val_pb_loss = total_val_pb_loss / len(X_val)
                current_target_vol = avg_val_vol

            # Saving the dic of best models
            if current_target_vol < self.best_valid_vol:
                self.best_valid_vol = current_target_vol
                self.best_model_weights = copy.deepcopy(self.state_dict())

            # Logs
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                tau_low, tau_high = self.tau_parameterAnnealer.get_params()
                if has_val:
                    print(f"Epoch {epoch+1}/{epochs} | T-Loss: {avg_train_loss:.4f} | V-Loss: {avg_val_loss:.4f} | "
                          f"T-Cov: {avg_train_cov:.3f} | V-Cov: {avg_val_cov:.3f} | "
                          f"T-Vol: {avg_train_vol:.4f} | V-Vol: {avg_val_vol:.2f} | "
                          f"BEST-Vol: {self.best_valid_vol:.4f} | tau_low: {tau_low:.2f} | tau_high: {tau_high:.2f} | "
                          f"T-pb-loss: {avg_train_pb_loss:.2f} | V-pb-loss: {avg_val_pb_loss:.2f} | "
                          )

                else:
                    print(f"Epoch {epoch+1}/{epochs} | T-Loss: {avg_train_loss:.4f} | "
                          f"T-Cov: {avg_train_cov:.3f} | "
                          f"T-Vol: {avg_train_vol:.4f} | "
                          f"BEST-Vol: {self.best_valid_vol:.4f} | tau_low: {tau_low:.2f} | tau_high: {tau_high:.2f} | "
                          f"T-pb-loss: {avg_train_pb_loss:.2f} "
                          )

        # Loading best weights in the end if requested
        if self.best_model_weights is not None and return_best:
            print(f"Entraînement terminé. Restauration des meilleurs poids avec un Volume Valide de : {self.best_valid_vol:.4f}")
            self.load_state_dict(self.best_model_weights)
            
        return self
    
    def fit_external_quantile(
            self, 
            quantile_model,
            X_train, 
            y_train, 
            batch_size = 300,
            **kwargs
            ): 
        
        frontiers_list = []

        with torch.no_grad():
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                frontiers_batch, _, _ = self.get_frontiers(X_batch, y_batch)
                frontiers_list.append(frontiers_batch.detach()) 
                # frontiers_list.append(frontiers_batch.detach().cpu()) 
            frontiers_train = torch.cat(frontiers_list, dim=0)
        
        self.external_quantile_model = quantile_model
        self.external_quantile_model.fit(X_train, frontiers_train.squeeze())
    
        self.external_quantile = True

    def fit_quantile(
            self, 
            X_train, 
            y_train, 
            tau, 
            X_val=None, 
            y_val=None, 
            epochs=1000, 
            lr=1e-3, 
            batch_size=32, 
            weight_decay=1e-4, 
            return_best=True, 
            print_every=1,
            restart=True
            ): 
        
        if restart:
            self.quantile_manager.reset_parameters()
        
        # Dataloaders preparation
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

        # Keep track of the best model
        best_target_loss = float('inf')
        self.best_quantile_model_weights = None

        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_coverage = 0.0
            total_train_pb_loss = 0.0
            
            for batch_X, batch_y in train_loader:
            
                ### Treat the quantiles
                optimizer.zero_grad()
            
                loss_tau, coverage = self.compute_pb_loss(batch_X, batch_y, tau, with_coverage=True)
                loss_pb =  loss_tau
                loss_pb.backward()
                
                epoch_coverage += coverage
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_train_pb_loss += loss_pb.item() * batch_X.size(0)
                
            avg_train_cov = epoch_coverage / len(X_train)
            avg_train_pb_loss = total_train_pb_loss / len(X_train)

            current_target_loss = avg_train_pb_loss

            # Validation if possible
            if has_val:
                self.eval()
                val_epoch_coverage = 0.0
                total_val_pb_loss = 0.0
                
                with torch.no_grad():
                    for v_batch_X, v_batch_y in val_loader:
                        
                        loss_pb, coverage = self.compute_pb_loss(v_batch_X, v_batch_y, tau, with_coverage=True)
                        total_val_pb_loss += loss_pb.item() * v_batch_X.size(0)
                        val_epoch_coverage += coverage
                        
                avg_val_cov = val_epoch_coverage / len(X_val)
                avg_val_pb_loss = total_val_pb_loss / len(X_val)

                current_target_loss = avg_val_pb_loss

            # Saving the dic of best models
            if current_target_loss < best_target_loss:
                best_target_loss = current_target_loss
                self.best_quantile_model_weights = copy.deepcopy(self.state_dict())

            # Logs
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                if has_val:
                    print(f"Epoch {epoch+1}/{epochs} | T-Loss: {avg_train_pb_loss:.4f} | V-Loss: {avg_val_pb_loss:.4f} | "
                          f"T-Cov: {avg_train_cov:.3f} | V-Cov: {avg_val_cov:.3f} | "
                          )
                else:
                    print(f"Epoch {epoch+1}/{epochs} | T-Loss: {avg_train_pb_loss:.4f} | "
                          f"T-Cov: {avg_train_cov:.3f} |"
                          )

        # Loading best weights in the end if requested
        if self.best_quantile_model_weights is not None and return_best:
            self.load_state_dict(self.best_quantile_model_weights)
            
        return self
   
    def compute_p(self, X, S):
        q_low, q_high = self.get_q_kernel(X)
        p = ((q_low < S) & (S < q_high)).float()
        return p 
    
    def compute_loss(self, X, y, tau):
        G, log_det_L_stacked, q = self.get_frontiers(X, y)
        q_detached = q.detach()
        d = y.shape[-1]

        p = self.compute_p(X, G) if self.num_iteration > self.tau_parameterAnnealer.warm_start_step else torch.ones_like(G)
        
        if self.loss_function == "only_quantile": 
            loss = (p * G).mean()
        elif self.loss_function == "log_volume":
            d = y.shape[-1]
            log_vol_penalty = torch.logsumexp(-log_det_L_stacked, dim=1, keepdim=True)
            log_G_term = (d / 2.0) * torch.log(torch.relu(G) + 1e-6)
            log_loss_volumes = log_vol_penalty + log_G_term
            safe_log_loss_volumes = torch.clamp(log_loss_volumes, max=85.0, min=-85.0)
            loss = (p * safe_log_loss_volumes).mean()
        elif self.loss_function == "full_volume":
            d = y.shape[-1]
            log_vol_penalty = torch.logsumexp(-log_det_L_stacked, dim=1, keepdim=True)
            log_G_term = (d / 2.0) * torch.log(torch.relu(G) + 1e-6)
            log_loss_volumes = log_vol_penalty + log_G_term
            safe_log_loss_volumes = torch.clamp(log_loss_volumes, max=85.0, min=-85.0)
            loss_volumes = torch.exp(safe_log_loss_volumes)
            loss = (p * loss_volumes).mean()
        elif self.loss_function == "NLL":
            d = y.shape[-1]
            log_vol_penalty = torch.logsumexp(-log_det_L_stacked, dim=1, keepdim=True)
            G_term = G
            log_loss_NLL = log_vol_penalty + G_term
            safe_log_loss_NLL = torch.clamp(log_loss_NLL, max=85.0, min=-85.0)
            loss = (p * safe_log_loss_NLL).mean()
        else:
            raise ValueError("The loss_function must be only_quantile or log_volume or full_volume or NLL.")
        
        with torch.no_grad():
            G_detached = G.detach()
            batch_coverage = (G_detached <= q_detached).float().mean() 
            
            scores = (G_detached / q_detached).squeeze()
            if scores.dim() == 0 or scores.size(0) < 2:
                valid_volume = torch.tensor(0.0, device=X.device)
            else:
                batch_q_alpha = torch.quantile(scores, tau)
                valid_q = q_detached * batch_q_alpha
                log_det_L_stacked, _ = self.get_det_L(X)

                log_vol_penalty = torch.logsumexp(-log_det_L_stacked, dim=1, keepdim=True)
                log_V_constant = math.log((math.pi ** (self.d / 2)) / math.gamma(self.d / 2 + 1))
                log_vol_k = log_V_constant + (self.d / 2) * torch.log(valid_q) + log_vol_penalty
                            
                valid_volume = torch.exp(log_vol_k).mean()
            
        return loss, batch_coverage, valid_volume

    def compute_pb_loss(self, X, y, tau, with_coverage=False):
        G, _, q = self.get_frontiers(X, y)

        G_sg = G.detach()

        loss_q = (tau * F.relu(G_sg - q) + (1 - tau) * F.relu(q - G_sg)).mean()
        
        if with_coverage:
            coverage = (G_sg <= q).float().sum()
            return loss_q, coverage
        return loss_q
    
    def compute_pb_kernel_loss(self, X, y):
        q_low, q_high = self.get_q_kernel(X)
        S, _, _ = self.get_frontiers(X, y)
        G_sg = S.detach()

        tau_low, tau_high = self.tau_parameterAnnealer.step()

        loss_q_low = (tau_low * F.relu(G_sg - q_low) + (1 - tau_low) * F.relu(q_low - G_sg)).mean()
        loss_q_high = (tau_high * F.relu(G_sg - q_high) + (1 - tau_high) * F.relu(q_high - G_sg)).mean()
        
        return loss_q_low, loss_q_high

