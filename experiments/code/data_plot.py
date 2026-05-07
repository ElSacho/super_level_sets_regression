import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import copy
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.lines import Line2D

def plot_1d_conditional_contours(
    data_generator,
    model,
    n_samples=50,
    res_points=100,
    plot_mu=True,
    f=None
):

    model.eval()

    xs = torch.linspace(-1, 1, 500)

    # --------------------------------------------------
    # 1. Génération des samples Y|X pour visualisation
    # --------------------------------------------------

    y_all = []
    samples_all = []
    x_samples_all = []

    for x_val in xs:

        x_tensor = x_val.view(1,1)

        samples = data_generator.generate_specific_y_given_x(
            x_tensor,
            n=n_samples
        )

        y_all.append(samples)

        samples_all.append(samples)
        x_samples_all.append(torch.full((n_samples,1), x_val))

    samples_all = torch.cat(samples_all).numpy()
    x_samples_all = torch.cat(x_samples_all).numpy()

    y_all = torch.cat(y_all, dim=0)

    y_min = y_all.min().item() - 1
    y_max = y_all.max().item() + 1

    # --------------------------------------------------
    # 2. Construction de la grille (X,Y)
    # --------------------------------------------------

    y_grid_vals = torch.linspace(y_min, y_max, res_points)

    Xg, Yg = torch.meshgrid(xs, y_grid_vals, indexing="ij")

    X_grid = Xg.reshape(-1,1)
    Y_grid = Yg.reshape(-1,1)

    # --------------------------------------------------
    # 3. Calcul du score S(x,y) en 1 ligne
    # --------------------------------------------------

    with torch.no_grad():
        # --- NOUVEAU : Appel à get_frontiers ---
        S_grid_raw, _, _ = model.get_frontiers(X_grid, Y_grid)
        S_grid = S_grid_raw.reshape(len(xs), res_points)

        # Récupération de q(x)
        # On utilise [-1] pour garantir qu'on prend bien q, même s'il y a la partition
        conformalize_out = model.call_conformalize(xs.unsqueeze(1))
        q_vals = conformalize_out[-1][:, 0]

    # --------------------------------------------------
    # 4. Région conforme
    # --------------------------------------------------

    q_grid = q_vals.unsqueeze(1).repeat(1, res_points)

    mask = (S_grid <= q_grid)

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------

    plt.figure(figsize=(6,5), dpi=120)

    X_plot, Y_plot = np.meshgrid(xs.numpy(), y_grid_vals.numpy(), indexing="ij")

    plt.contourf(
        X_plot,
        Y_plot,
        mask.numpy(),
        levels=[0.5,1],
        alpha=0.3
    )

    plt.contour(
        X_plot,
        Y_plot,
        (S_grid - q_grid).numpy(),
        levels=[0],
        linewidths=2
    )

    plt.scatter(
        x_samples_all,
        samples_all,
        s=3,
        alpha=0.25
    )

    # --------------------------------------------------
    # 6. Courbes mu_k(x)
    # --------------------------------------------------

    if plot_mu:

        with torch.no_grad():
            # Récupération de mu (toujours le premier élément)
            mu_vals = model.call_conformalize(xs.unsqueeze(1))[0]

        for k in range(model.K):

            plt.plot(
                xs.numpy(),
                mu_vals[:,k,0].numpy(),
                linewidth=2,
                color="red"
            )

    if f is not None:

        mu_vals = f(xs.unsqueeze(1))

        plt.plot(
            xs.numpy(),
            mu_vals[:,0].numpy(),
            linewidth=2,
            color="green"
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("1D conditional conformal region")

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.show()

def plot_1X(model, tau, X_train, Y_train, res_points=150):

    model.eval()

    with torch.no_grad():
        # 1. Récupération du seuil calibré q_val et de la partition (pour le titre)
        # On utilise call_conformalize ici uniquement pour récupérer le seuil final calibré
        _, _, partition, q_conf = model.call_conformalize(X_train)
        q_val = q_conf[0, 0].item()
        
        # --- NOUVEAU : Calcul des scores d'entraînement en 1 ligne ---
        S_y, _, _ = model.get_frontiers(X_train, Y_train)
        S_y = S_y.squeeze()
            
        inside_mask = (S_y <= q_val)
        points_inside = Y_train[inside_mask]
        points_outside = Y_train[~inside_mask]
        empirical_coverage = inside_mask.float().mean().item()
        
        print(f"\nCouverture empirique : {empirical_coverage*100:.1f}% (Cible: {tau*100:.0f}%)")

        # 2. Création de la grille d'affichage
        y1_min, y1_max = Y_train[:, 0].min().item() - 1, Y_train[:, 0].max().item() + 1
        y2_min, y2_max = Y_train[:, 1].min().item() - 1, Y_train[:, 1].max().item() + 1
        
        y1_grid, y2_grid = torch.meshgrid(
            torch.linspace(y1_min, y1_max, res_points),
            torch.linspace(y2_min, y2_max, res_points),
            indexing='ij'
        )
        Y_grid = torch.stack([y1_grid.flatten(), y2_grid.flatten()], dim=1)
        
        # On répète la première valeur de X_train pour la grille conditionnelle
        X_grid = X_train[0:1].repeat(Y_grid.shape[0], 1)
        
        # 3. Inférence sur la grille
        # --- NOUVEAU : Calcul des scores de la grille en 1 ligne ---
        S_grid_raw, _, _ = model.get_frontiers(X_grid, Y_grid)
        
        S_grid = S_grid_raw.reshape(res_points, res_points)

    # 4. Tracé
    plt.figure(figsize=(10, 8))

    # Coloration de l'intérieur de la région
    plt.contourf(y1_grid.numpy(), y2_grid.numpy(), S_grid.numpy(), 
                levels=[-1e6, q_val], colors=['dodgerblue'], alpha=0.2)

    contour = plt.contour(y1_grid.numpy(), y2_grid.numpy(), S_grid.numpy(), 
                        levels=[q_val], colors='black', linewidths=3)

    plt.scatter(points_inside[:, 0].numpy(), points_inside[:, 1].numpy(), 
                c='blue', alpha=0.5, label='Inlier (Conservé)', s=15)
    plt.scatter(points_outside[:, 0].numpy(), points_outside[:, 1].numpy(), 
                c='red', alpha=0.5, label='Outlier (Rejeté)', s=15)

    titre = f"Union de {model.K} Normalizing Flows (Softmin pondéré)" if partition is not None else f"Union de {model.K} Normalizing Flows (Softmin standard)"
    plt.title(f"{titre}\nCible: {tau*100:.0f}% | Obtenu: {empirical_coverage*100:.1f}%")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(y1_min, y1_max)
    plt.ylim(y2_min, y2_max)
    plt.show()

def plot_2d_conditional_contours_gaussian(data_generator, model, xs=[-0.8, -0.4, 0.0, 0.4, 0.8], n_samples=500):
    n_plots = len(xs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=100)
    
    if n_plots == 1:
        axes = [axes]

    for ax, x_val in zip(axes, xs):
        # 1. Préparation du tenseur X
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        
        # 2. Génération des échantillons réels
        with torch.no_grad():
            samples = data_generator.generate_specific_y_given_x(x_tensor, n=n_samples)
        samples_np = samples.cpu().numpy()
        if samples_np.shape[0] != n_samples: samples_np = samples_np.T

        # 3. Récupération des paramètres de la distribution par le modèle
        with torch.no_grad():
            center, sigma = model.get_distribution(x_tensor)
            # Calcul de la racine carrée de la matrice de covariance (pour l'ellipse)
            L, Q = torch.linalg.eigh(sigma)
            L_sqrt = torch.diag_embed(torch.sqrt(L.clamp(min=1e-12)))
            sigma_sqrt = (Q @ L_sqrt @ Q.transpose(-2, -1)).squeeze(0).cpu().numpy()
            
            mu = center.squeeze(0).cpu().numpy()
            
            # Métriques pour le titre
            coverage = model.get_coverage(x_tensor, samples) * 100
            # On suppose que get_average_volume accepte le tenseur x
            volume = model.get_average_volume(x_tensor)

        # 4. Calcul de l'ellipse théorique (C_alpha)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_points = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        # On multiplie par le seuil q_alpha
        ellipse_points = mu[:, None] + sigma_sqrt @ (model.q_alpha * circle_points)

        # 5. Plot
        # Échantillons
        ax.scatter(samples_np[:, 0], samples_np[:, 1], c='grey', s=10, alpha=0.4, label='Samples')
        
        # Contour du modèle (Ellipse)
        ax.plot(ellipse_points[0, :], ellipse_points[1, :], color='black', linewidth=2, label=r'$C_\alpha(X)$')
        
        # Centre de la gaussienne
        ax.scatter(mu[0], mu[1], color='red', marker='x', s=50, zorder=3)

        # 6. Cosmétique
        ax.set_title(f"X = {x_val:.2f}\nCov: {coverage:.1f}% | Vol: {volume:.2f}")
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Ajustement dynamique des axes pour que l'ellipse soit bien visible
        margin = 1.5
        ax.set_xlim(samples_np[:, 0].min() - margin, samples_np[:, 0].max() + margin)
        ax.set_ylim(samples_np[:, 1].min() - margin, samples_np[:, 1].max() + margin)

    plt.tight_layout()
    plt.show()

def plot_2d_conditional_contours(data_generator, model, xs=[-0.8, -0.4, 0.0, 0.4, 0.8], n_samples=500, res_points=100):
    model.eval()

    n_plots = len(xs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), dpi=100)
    if n_plots == 1:
        axes = [axes]

    for ax, x_val in zip(axes, xs):
        # 1. Génération des points
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        samples = data_generator.generate_specific_y_given_x(x_tensor, n=n_samples)
        samples_np = samples.numpy()

        # 2. Définition de la grille
        y1_min, y1_max = samples_np[:, 0].min() - 1, samples_np[:, 0].max() + 1
        y2_min, y2_max = samples_np[:, 1].min() - 1, samples_np[:, 1].max() + 1
        
        y1_grid_vals = torch.linspace(y1_min, y1_max, res_points)
        y2_grid_vals = torch.linspace(y2_min, y2_max, res_points)
        y1_grid, y2_grid = torch.meshgrid(y1_grid_vals, y2_grid_vals, indexing='ij')
        
        Y_grid = torch.stack([y1_grid.flatten(), y2_grid.flatten()], dim=1)

        with torch.no_grad():
            # --- NOUVEAU : Récupération du seuil calibré q_val ---
            # On récupère le dernier élément renvoyé par call_conformalize (qui est q_conf)
            conformalize_out = model.call_conformalize(x_tensor)
            q_val = conformalize_out[-1][0, 0].item()
            
            # --- NOUVEAU : SOFTMIN SUR LA GRILLE (en 2 lignes) ---
            X_grid = x_tensor.repeat(Y_grid.shape[0], 1)
            S_grid_raw, _, _ = model.get_frontiers(X_grid, Y_grid)
            S_grid = S_grid_raw.reshape(res_points, res_points)
            
            # --- NOUVEAU : SOFTMIN SUR LES POINTS (Pour calculer la couverture) ---
            X_samples = x_tensor.repeat(n_samples, 1)
            S_samples_raw, _, _ = model.get_frontiers(X_samples, samples)
            S_samples = S_samples_raw.squeeze(-1) # [n_samples]
            
            inliers_mask = (S_samples <= q_val).numpy()
            
            # Récupération du Volume
            volume = model.compute_volume(x_tensor).item()

        # 3. Calcul de la couverture
        coverage = inliers_mask.mean() * 100
        
        # 4. Tracé 2D
        # Coloration de l'intérieur de la région
        ax.contourf(y1_grid.numpy(), y2_grid.numpy(), S_grid.numpy(), 
                    levels=[-1e6, q_val], colors=['dodgerblue'], alpha=0.2)
             
        ax.scatter(samples_np[inliers_mask, 0], samples_np[inliers_mask, 1], c='blue', s=10, alpha=0.5, label='Inlier')

        ax.scatter(samples_np[~inliers_mask, 0], samples_np[~inliers_mask, 1], c='red', s=10, alpha=0.5, label='Outlier')
        
        ax.contour(y1_grid.numpy(), y2_grid.numpy(), S_grid.numpy(), 
                   levels=[q_val], colors='black', linewidths=2)
        
        ax.set_title(f"X = {x_val:.2f} | Cov: {coverage:.1f}% | Vol: {volume:.1f}")
        ax.set_xlabel("Y1")
        ax.set_ylabel("Y2")
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_chdr_with_samples_gaussian(model, data_generator, x_range=(-1, 1), num_slices=12, n_samples=100, name="gaussian"):
    FS = 18

    fig = plt.figure(figsize=(8, 8), dpi=120) 
    ax = fig.add_subplot(111, projection='3d')
    
    xs = np.linspace(x_range[0], x_range[1], num_slices)
    
    levels = [model.q_alpha]
    colors = ['black'] 
    linewidths = [2]
    
    for x_val in xs:
        x_tensor = np.array([x_val]) 
        center, sigma = model.get_distribution(torch.tensor(x_tensor, dtype=torch.float32).unsqueeze(0))
        L, Q = torch.linalg.eigh(sigma)
        L_sqrt = torch.diag_embed(torch.sqrt(L.clamp(min=1e-12)))
        sigma_sqrt = Q @ L_sqrt @ Q.transpose(-2, -1)
        theta = np.linspace(0, 2*np.pi, 100)
        circle_points = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        for r, col, lw in zip(levels, colors, linewidths):
            ellipse_points = center[0][:, None].detach().cpu().numpy() + sigma_sqrt.detach().cpu().numpy() @ (r * circle_points)
            ellipse_points = ellipse_points.reshape(2, 100)
            ax.plot(np.full_like(ellipse_points[0, :], x_val), ellipse_points[0, :], ellipse_points[1, :], color=col, linewidth=lw, alpha=0.8, zorder=2)
            
        samples = data_generator.generate_specific_y_given_x(torch.tensor(x_tensor, dtype=torch.float32).unsqueeze(0), n=n_samples)
        if hasattr(samples, 'detach'): samples = samples.detach().cpu().numpy()
        if samples.shape[0] != n_samples and samples.shape[1] == n_samples: samples = samples.T
        ax.scatter(np.full(n_samples, x_val), samples[:, 0], samples[:, 1], c='grey', marker='o', s=5, alpha=0.3, zorder=1)

    ax.set_xlabel(r'$X$', fontsize=FS, labelpad=15)
    ax.set_ylabel(r'$Y_1$', fontsize=FS, labelpad=15)
    
    ax.text(0.75, 16, 5, r'$Y_2$', fontsize=FS, fontweight='bold', ha='center')

    ax.set_xlim(x_range[0], x_range[1])
    z_limit = 7
    ax.set_zlim(-z_limit, z_limit)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(axis='both', which='major', labelsize=FS)

    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label=r'$C_\alpha(X)$'),
        Line2D([0], [0], marker='o', color='w', label=r'Samples from $\mathbb{P}_{Y|X}$',
               markerfacecolor='grey', markersize=8, alpha=0.5)
    ]
    ax.legend(handles=legend_elements, fontsize=FS, loc='upper left')

    ax.view_init(elev=20, azim=-60)
    

    plt.subplots_adjust(left=0.0, right=0.95, bottom=0.05, top=0.95)
    plt.show()


def plot_samples_with_contours_multi_flow(data_generator, model, tau=0.9, x_range=(-1, 1), num_slices=8, n_samples=200):
    model.eval()

    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    xs = np.linspace(x_range[0], x_range[1], num_slices)

    for x_val in xs:

        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        samples = data_generator.generate_specific_y_given_x(x_tensor, n=n_samples)
        samples_np = samples.numpy()

        y1_min, y1_max = samples[:, 0].min().item() - 1, samples[:, 0].max().item() + 1
        y2_min, y2_max = samples[:, 1].min().item() - 1, samples[:, 1].max().item() + 1

        y1_grid_vals = np.linspace(y1_min, y1_max, 100)
        y2_grid_vals = np.linspace(y2_min, y2_max, 100)

        Y1, Y2 = np.meshgrid(y1_grid_vals, y2_grid_vals)

        y1_tensor = torch.tensor(Y1.flatten(), dtype=torch.float32)
        y2_tensor = torch.tensor(Y2.flatten(), dtype=torch.float32)

        Y_grid = torch.stack([y1_tensor, y2_tensor], dim=1)

        x_scatter = np.full(n_samples, x_val)

        ax.scatter(
            x_scatter,
            samples_np[:, 0],
            samples_np[:, 1],
            c="grey",
            s=8,
            alpha=0.5,
            zorder=1
        )

        with torch.no_grad():
            # --- NOUVEAU : Récupération du seuil calibré q_val ---
            # Optimisation: on le calcule sur x_tensor (1 point) au lieu de X_grid (10 000 points)
            conformalize_out = model.call_conformalize(x_tensor)
            q_val = conformalize_out[-1][0, 0].item()

            # --- NOUVEAU : Calcul des scores de la grille en 2 lignes ---
            X_grid = x_tensor.repeat(Y_grid.shape[0], 1)
            S_grid_raw, _, _ = model.get_frontiers(X_grid, Y_grid)
            
            # On redimensionne directement le score brut pour l'affichage
            S_grid_2d = S_grid_raw.reshape(Y1.shape).numpy()

        if S_grid_2d.min() <= q_val <= S_grid_2d.max():

            fig_dummy, ax_dummy = plt.subplots()

            cs = ax_dummy.contour(Y1, Y2, S_grid_2d, levels=[q_val])

            paths = cs.collections[0].get_paths() if hasattr(cs, 'collections') else cs.get_paths()

            for path in paths:

                vertices = path.vertices

                x_line = np.full(vertices.shape[0], x_val)
                y1_line = vertices[:, 0]
                y2_line = vertices[:, 1]

                ax.plot(
                    x_line,
                    y1_line,
                    y2_line,
                    color='red',
                    linewidth=2.5,
                    zorder=10
                )

            plt.close(fig_dummy)

        else:
            print(f"Attention: quantile {q_val:.2f} hors grille pour x={x_val:.2f}")

    ax.set_xlabel('X', fontsize=18, labelpad=10)
    ax.set_ylabel('Y1', fontsize=18, labelpad=10)
    ax.set_zlabel('Y2', fontsize=18, labelpad=10)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 6)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.view_init(elev=20, azim=-67)

    plt.tight_layout()
    plt.title(f"Régions de confiance conditionnelles {tau*100:.0f}%")
    plt.show()

def plot_combined_conditional_contours(data_generator, gaussian_model, flow_model, 
                                      xs=[-0.8, -0.4, 0.0, 0.4, 0.8], n_samples=500):
    """
    Superpose les contours de confiance du modèle Gaussien et du Normalizing Flow.
    """
    flow_model.eval()
    
    n_plots = len(xs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=100)
    
    if n_plots == 1:
        axes = [axes]

    for ax, x_val in zip(axes, xs):
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        
        # 1. Génération des échantillons réels (vérité terrain)
        with torch.no_grad():
            samples = data_generator.generate_specific_y_given_x(x_tensor, n=n_samples)
        samples_np = samples.cpu().numpy()
        if samples_np.shape[0] != n_samples: samples_np = samples_np.T

        # --- PARTIE GAUSSIENNE ---
        with torch.no_grad():
            center, sigma = gaussian_model.get_distribution(x_tensor)
            # Décomposition pour l'ellipse
            L_eig, Q_eig = torch.linalg.eigh(sigma)
            L_sqrt = torch.diag_embed(torch.sqrt(L_eig.clamp(min=1e-12)))
            sigma_sqrt = (Q_eig @ L_sqrt @ Q_eig.transpose(-2, -1)).squeeze(0).cpu().numpy()
            mu_gauss = center.squeeze(0).cpu().numpy()
            
            # Calcul couverture et ellipse
            cov_gauss = gaussian_model.get_coverage(x_tensor, samples) * 100
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
            ellipse_pts = mu_gauss[:, None] + sigma_sqrt @ (gaussian_model.q_alpha * circle)

        # --- PARTIE FLOW ---
        y1_min, y1_max = samples_np[:, 0].min() - 1, samples_np[:, 0].max() + 1
        y2_min, y2_max = samples_np[:, 1].min() - 1, samples_np[:, 1].max() + 1
        
        y1_g, y2_g = np.meshgrid(np.linspace(y1_min, y1_max, 100), np.linspace(y2_min, y2_max, 100))
        Y_grid = torch.tensor(np.stack([y1_g.ravel(), y2_g.ravel()], axis=1), dtype=torch.float32)

        with torch.no_grad():
            # --- NOUVEAU : Récupération propre de q_val sur 1 point ---
            conformalize_out = flow_model.call_conformalize(x_tensor)
            q_val_f = conformalize_out[-1][0, 0].item()
            
            # --- NOUVEAU : Calcul sur la grille ---
            X_grid = x_tensor.repeat(Y_grid.shape[0], 1)
            S_grid_raw, _, _ = flow_model.get_frontiers(X_grid, Y_grid)
            S_grid = S_grid_raw.reshape(100, 100)
            
            # --- NOUVEAU : Couverture empirique du Flow ---
            X_samples = x_tensor.repeat(n_samples, 1)
            S_samples_raw, _, _ = flow_model.get_frontiers(X_samples, samples)
            cov_flow = (S_samples_raw.squeeze(-1) <= q_val_f).float().mean().item() * 100

        # --- PLOT ---
        # 1. Samples
        ax.scatter(samples_np[:, 0], samples_np[:, 1], c='lightgrey', s=10, alpha=0.5, label='Samples')
        
        # 2. Contour Gaussien (Pointillé)
        ax.plot(ellipse_pts[0, :], ellipse_pts[1, :], color='red', linestyle='--', linewidth=2, label='Gaussian')
        
        # 3. Contour Flow (Plein)
        contour_flow = ax.contour(y1_g, y2_g, S_grid.numpy(), levels=[q_val_f], colors='blue', linewidths=2)
        # Astuce pour la légende du contour
        ax.plot([], [], color='blue', linewidth=2, label='Flow-based')

        # Cosmétique
        ax.set_title(f"X = {x_val:.2f}\n"
                     f"Gauss: {cov_gauss:.1f}% | Flow: {cov_flow:.1f}%")
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(True, linestyle=':', alpha=0.6)
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def plot_combined_conditional_contours_multiflow(data_generator, gaussian_model, flow_model, 
                                                      n_plots=5, idx_chosen=None, res_points=100,
                                                      xs=[-0.8, -0.4, 0.0, 0.4, 0.8], n_samples=500
                                                      ):
    """
    Superpose les contours du modèle Gaussien et du Multi-Normalizing Flow (Softmin)
    pour des points individuels du set de test.
    """
    flow_model.eval()

    n_plots = len(xs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=100)
    
    if n_plots == 1:
        axes = [axes]

    for ax, x_val in zip(axes, xs):
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        
        # 1. Génération des échantillons réels (vérité terrain)
        with torch.no_grad():
            samples = data_generator.generate_specific_y_given_x(x_tensor, n=n_samples)
        samples_np = samples.cpu().numpy()
        if samples_np.shape[0] != n_samples: samples_np = samples_np.T
        
        # --- PARTIE GAUSSIENNE ---
        with torch.no_grad():
            center, sigma = gaussian_model.get_distribution(x_tensor)
            
            # Décomposition pour l'ellipse
            L_eig, Q_eig = torch.linalg.eigh(sigma)
            L_sqrt = torch.diag_embed(torch.sqrt(L_eig.clamp(min=1e-12)))
            sigma_sqrt = (Q_eig @ L_sqrt @ Q_eig.transpose(-2, -1)).squeeze(0).cpu().numpy()
            mu_gauss = center.squeeze(0).cpu().numpy()
            
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
            ellipse_pts = mu_gauss[:, None] + sigma_sqrt @ (gaussian_model.q_alpha * circle)

        # --- LIMITES DE GRILLE (Basées sur l'ellipse Gaussienne) ---
        # On utilise l'écart-type Gaussien pour centrer la fenêtre correctement
        std_devs = np.sqrt(np.diag(sigma.squeeze(0).cpu().numpy()))
        margin = max(gaussian_model.q_alpha * 1.5, 3.0) 
        
        y1_min, y1_max = samples_np[:, 0].min() - 1, samples_np[:, 0].max() + 1
        y2_min, y2_max = samples_np[:, 1].min() - 1, samples_np[:, 1].max() + 1
        
        # La grille initiale utilise 100 points, on maintient cette valeur
        y1_g, y2_g = np.meshgrid(np.linspace(y1_min, y1_max, 100), np.linspace(y2_min, y2_max, 100))
        Y_grid = torch.tensor(np.stack([y1_g.ravel(), y2_g.ravel()], axis=1), dtype=torch.float32)

        with torch.no_grad():
            # --- NOUVEAU : Appel unique 1 point ---
            conformalize_out = flow_model.call_conformalize(x_tensor)
            q_val_f = conformalize_out[-1][0, 0].item()

            # --- NOUVEAU : Calcul sur la grille ---
            X_grid = x_tensor.repeat(Y_grid.shape[0], 1)
            S_grid_raw, _, _ = flow_model.get_frontiers(X_grid, Y_grid)
            S_grid = S_grid_raw.reshape(100, 100)

        # --- PLOT ---
        # 1. Remplissage et contour Flow
        contour_flow = ax.contour(y1_g, y2_g, S_grid.numpy(), levels=[q_val_f], colors='blue', linewidths=2)
        
        # 2. Contour Gaussien (Pointillé rouge)
        ax.plot(ellipse_pts[0, :], ellipse_pts[1, :], color='red', linestyle='--', linewidth=2, label='Gaussian')
        
        # Astuce pour la légende Flow
        ax.plot([], [], color='blue', linewidth=2, label='Multi-Flow')

        vol_gaussian = gaussian_model.get_average_volume(x_tensor)
        vol_level_set = flow_model.compute_average_volume(x_tensor).item()

        ax.scatter(samples_np[:, 0], samples_np[:, 1], c='lightgrey', s=10, alpha=0.5, label='Samples')

        # Cosmétique
        ax.set_title(f"||X|| = {torch.norm(x_tensor).item():.2f} \n Vol Gaussian {vol_gaussian:.2f} | Vol LevelSet {vol_level_set:.2f}")
            
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

def plot_combined_conditional_contours_test_multiflow(X_test, Y_test, gaussian_model, flow_model, 
                                                      n_plots=5, idx_chosen=None, res_points=100):
    """
    Superpose les contours du modèle Gaussien et du Multi-Normalizing Flow (Softmin)
    pour des points individuels du set de test.
    """
    flow_model.eval()
    
    # Selection of indices
    if idx_chosen is None:
        idx_chosen = torch.randperm(len(X_test))[:n_plots]
    else:
        n_plots = len(idx_chosen)
        
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=100)
    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, idx_chosen):
        x_tensor = X_test[idx:idx+1]
        y_tensor = Y_test[idx:idx+1]
        y_np = y_tensor.cpu().numpy()[0]
        
        # --- PARTIE GAUSSIENNE ---
        with torch.no_grad():
            center, sigma = gaussian_model.get_distribution(x_tensor)
            
            # Décomposition pour l'ellipse
            L_eig, Q_eig = torch.linalg.eigh(sigma)
            L_sqrt = torch.diag_embed(torch.sqrt(L_eig.clamp(min=1e-12)))
            sigma_sqrt = (Q_eig @ L_sqrt @ Q_eig.transpose(-2, -1)).squeeze(0).cpu().numpy()
            mu_gauss = center.squeeze(0).cpu().numpy()
            
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
            ellipse_pts = mu_gauss[:, None] + sigma_sqrt @ (gaussian_model.q_alpha * circle)

        # --- LIMITES DE GRILLE (Basées sur l'ellipse Gaussienne) ---
        # On utilise l'écart-type Gaussien pour centrer la fenêtre correctement
        std_devs = np.sqrt(np.diag(sigma.squeeze(0).cpu().numpy()))
        margin = max(gaussian_model.q_alpha * 1.5, 3.0) 
        
        y1_min = mu_gauss[0] - margin * std_devs[0]
        y1_max = mu_gauss[0] + margin * std_devs[0]
        y2_min = mu_gauss[1] - margin * std_devs[1]
        y2_max = mu_gauss[1] + margin * std_devs[1]

        # --- PARTIE MULTI-FLOW ---
        y1_grid_vals = torch.linspace(y1_min, y1_max, res_points)
        y2_grid_vals = torch.linspace(y2_min, y2_max, res_points)
        y1_grid, y2_grid = torch.meshgrid(y1_grid_vals, y2_grid_vals, indexing='ij')
        
        Y_grid = torch.stack([y1_grid.flatten(), y2_grid.flatten()], dim=1).to(x_tensor.device)

        with torch.no_grad():
            # --- NOUVEAU : Appel unique 1 point ---
            conformalize_out = flow_model.call_conformalize(x_tensor)
            q_val_f = conformalize_out[-1][0, 0].item()

            # --- NOUVEAU : Calcul sur la grille ---
            X_grid = x_tensor.repeat(Y_grid.shape[0], 1)
            S_grid_raw, _, _ = flow_model.get_frontiers(X_grid, Y_grid)
            S_grid = S_grid_raw.reshape(res_points, res_points)

        # --- PLOT ---
        # 1. Remplissage et contour Flow
        ax.contourf(y1_grid.numpy(), y2_grid.numpy(), S_grid.cpu().numpy(), 
                    levels=[-1e6, q_val_f], colors=['dodgerblue'], alpha=0.2)
        ax.contour(y1_grid.numpy(), y2_grid.numpy(), S_grid.cpu().numpy(), 
                   levels=[q_val_f], colors='blue', linewidths=2)
        
        # 2. Contour Gaussien (Pointillé rouge)
        ax.plot(ellipse_pts[0, :], ellipse_pts[1, :], color='red', linestyle='--', linewidth=2, label='Gaussian')
        
        # 3. Point de test (Ground Truth) au premier plan
        ax.scatter(y_np[0], y_np[1], c='black', marker='*', s=150, zorder=5, label='True Y')
        
        # Astuce pour la légende Flow
        ax.plot([], [], color='blue', linewidth=2, label='Multi-Flow')

        vol_gaussian = gaussian_model.get_average_volume(x_tensor)
        vol_level_set = flow_model.compute_average_volume(x_tensor).item()

        # Cosmétique
        if x_tensor.shape[1] == 1:
            ax.set_title(f"Test Idx: {idx} | X = {x_tensor[0, 0].item():.2f}")
        else:
            ax.set_title(f"Test Idx: {idx} | ||X|| = {torch.norm(x_tensor).item():.2f} \n Vol Gaussian {vol_gaussian:.2f} | Vol LevelSet {vol_level_set:.2f}")
            
        ax.set_xlabel(r"$Y_1$")
        ax.set_ylabel(r"$Y_2$")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()