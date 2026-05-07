import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
from torch.utils.data import DataLoader, TensorDataset


def generate_star(N, dim_y=2):
    if dim_y < 2:
        raise ValueError("Star shape requires at least 2 dimensions.")
    
    # Angles for the 5 outer points and 5 inner points of a star
    angles = np.linspace(0, 2 * np.pi, 11)
    r_outer = 2.0
    r_inner = 0.8
    
    # Alternate radii to create the star points
    radii = np.ones_like(angles)
    radii[::2] = r_outer
    radii[1::2] = r_inner
    
    # Convert polar to cartesian
    pts_x = radii * np.cos(angles)
    pts_y = radii * np.sin(angles)
    
    # Linear interpolation to get N points along these edges
    t = np.linspace(0, 10, N)
    star_x = np.interp(t, np.arange(11), pts_x)
    star_y = np.interp(t, np.arange(11), pts_y)
    
    # Add a bit of noise so it's not a perfect wireframe
    star_data = np.stack([star_x, star_y], axis=1)
    star_tensor = torch.from_numpy(star_data).float()
    star_tensor += torch.randn_like(star_tensor) * 0.05 # jitter
    
    # If dim_y > 2, pad with zeros or noise
    if dim_y > 2:
        extra = torch.zeros(N, dim_y - 2)
        star_tensor = torch.cat([star_tensor, extra], dim=1)
        
    return star_tensor

def generate_full_star(N, dim_y=2):
    points = []
    # On génère un surplus de points car on va en rejeter une partie
    batch_size = N * 3 
    
    while len(points) < N:
        # 1. Générer des points aléatoires dans un carré [-R, R]
        candidate_points = torch.empty(batch_size, 2).uniform_(-2, 2)
        
        # 2. Convertir en coordonnées polaires
        r = torch.sqrt(candidate_points[:, 0]**2 + candidate_points[:, 1]**2)
        theta = torch.atan2(candidate_points[:, 1], candidate_points[:, 0])
        
        # 3. Formule de l'étoile à 5 branches (seuil de rayon variable)
        # La fonction cos(5*theta/2) crée l'oscillation des branches
        # On utilise une approximation par ligne brisée pour des bords droits
        star_radius = 1.0 + 0.8 * torch.cos(5 * theta / 2).abs() 
        
        # Note : Pour une étoile parfaite, on utilise souvent la fonction :
        # r < R * cos(pi/n) / cos((theta % (2*pi/n)) - pi/n)
        
        # 4. Garder les points à l'intérieur du rayon de l'étoile
        mask = r < star_radius
        accepted = candidate_points[mask]
        points.append(accepted)
        
    # Concaténer et tronquer à exactement N points
    Y_star = torch.cat(points, dim=0)[:N]
    
    # Gérer les dimensions supplémentaires si nécessaire
    if dim_y > 2:
        extra = torch.randn(N, dim_y - 2) * 0.01
        Y_star = torch.cat([Y_star, extra], dim=1)
        
    return Y_star

def sample_annulus(n, r_min=0.5, r_max=1.0):
    theta = 2 * np.pi * np.random.rand(n)
    
    u = np.random.rand(n)
    r = np.sqrt(u * (r_max**2 - r_min**2) + r_min**2)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return torch.as_tensor(np.stack((x, y), axis=1), dtype=torch.float32)


def generate_gaussian_mixture(N):
    # Mixture weights (must sum to 1)
    weights = np.array([0.5, 0.5])
    
    # Means of the 4 Gaussians (2D)
    means = np.array([
        [-10, -10],
        [10, 10],
    ])
    
    # Covariance matrices
    covs = [
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
    ]

    # Choose which Gaussian each sample comes from
    components = np.random.choice(2, size=N, p=weights)

    Y_train = np.zeros((N, 2))

    for k in range(2):
        idx = components == k
        n_k = np.sum(idx)
        if n_k > 0:
            Y_train[idx] = np.random.multivariate_normal(means[k], covs[k], n_k)

    return torch.as_tensor(Y_train, dtype=torch.float32)





class Generator2D:
    def __init__(self, f, matrix_transform, noise_type='gaussian', noise_std=1.0):
        self.f = f
        self.matrix_transform = matrix_transform
        self.noise_type = noise_type
        self.noise_std = noise_std

    def _get_noise(self, n):
        if self.noise_type == 'gaussian':
            noise = torch.randn(n, 2)
        elif self.noise_type == 'uniform':
            noise = torch.rand(n, 2) * 2 - 1
        elif self.noise_type == 'exponential':
            noise = torch.distributions.Exponential(rate=1.0).sample((n, 2))
        else:
            raise ValueError(f"Type inconnu: {self.noise_type}")
        return (noise * self.noise_std).unsqueeze(2)

    def generate(self, n):
        x = 2 * torch.rand(n, 1) - 1
        fx = self.f(x)
        A_x = self.matrix_transform(x)
        noise = self._get_noise(n)
        correlated_noise = torch.bmm(A_x, noise).squeeze(2)
        y = fx + correlated_noise
        return x, y

    def generate_specific_y_given_x(self, x_tensor, n=1):
        x_repeated = x_tensor.repeat_interleave(n, dim=0)
        fx = self.f(x_repeated)
        A_x = self.matrix_transform(x_repeated)
        noise = self._get_noise(x_repeated.shape[0])
        correlated_noise = torch.bmm(A_x, noise).squeeze(2)
        y_flat = fx + correlated_noise
        return y_flat.view(n, 2)

def strange_matrix_transform(x):
    n = x.shape[0]
    matrices = torch.eye(2).unsqueeze(0).repeat(n, 1, 1)
    matrices[:, 0, 0] = x.squeeze(-1)**2 + 0.5 
    matrices[:, 0, 1] = torch.sin(x.squeeze(-1) * 2)
    matrices[:, 1, 1] = torch.abs(x.squeeze(-1)) + 0.2
    return matrices

def circle_f(x):
    return torch.cat([torch.sin(x*3), torch.cos(x*3)], dim=1)


def strange_matrix_transform_nd_wrapper(dim_y):
    def strange_matrix_transform_nd(x: torch.Tensor) -> torch.Tensor:
        """
        Generalizes the 2D strange_matrix_transform to an arbitrary dim_y.
        
        Args:
            x: Tensor of shape (n, 1) or (n,).
            dim_y: The desired output dimensionality (k).
            
        Returns:
            Tensor of shape (n, dim_y, dim_y).
        """
        n = x.shape[0]
        
        
        x_val = x.view(n) 
        matrices = torch.eye(dim_y, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(n, 1, 1)
        
        
        for i in range(dim_y):
            if i % 2 == 0:
                matrices[:, i, i] = x_val**2 + 0.5 + (0.1 * i)
            else:
                matrices[:, i, i] = torch.abs(x_val) + 0.2 + (0.1 * i)
                
        for i in range(dim_y - 1):
            matrices[:, i, i+1] = torch.sin(x_val * (2.0 + i))
            
        return matrices
    
    return strange_matrix_transform_nd
           

