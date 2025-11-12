import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.random as random
from functools import partial
from typing import Callable, Union
from collections import namedtuple
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

from model import model
from methods import VGD
from experiment import experiment, diagnostic_experiment

def predictive_posterior_distribution_k(
    thetas, 
    x, 
    model, 
    sigma, 
    key=random.PRNGKey(0),
    num_noise_samples_per_theta: int = 5  
    ):

    pred_key, key = random.split(key)

    predictive_means = jax.vmap(model, in_axes=(0, None))(thetas, x)

    num_thetas, num_x_points = predictive_means.shape
    K = num_noise_samples_per_theta # K

    noise_shape = (num_thetas, K, 1)
    noises = random.normal(pred_key, shape=noise_shape) * sigma

    predictive_means_expanded = predictive_means[:, None, :] 

    posterior_predictive_samples_skn = predictive_means_expanded + noises

    total_samples = num_thetas * K
    final_samples = posterior_predictive_samples_skn.reshape(total_samples, num_x_points)

    return final_samples

def plot_shaded_region_predictive(ax, experiment: experiment, particles, color, intervals=[50, 80, 90]):
    x = np.sort(jnp.array(experiment.data)[0, :])
    ax.scatter(*experiment.data, marker='.', s=0.7, color='black', alpha=0.6, zorder=0)
    fn = experiment.fn
    sigma = experiment.sigma

    samples = predictive_posterior_distribution_k(particles, x, fn, sigma=sigma)
    mean_curve = jnp.mean(samples, axis=0)

    ax.plot(x, mean_curve, color=color, linewidth=2, zorder=3)
    ax.tick_params(axis='both', labelsize=16)

    intervals.sort(reverse=True)
    
    for p in intervals:
        lower_percentile = (100 - p) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = jnp.percentile(samples, lower_percentile, axis=0)
        upper_bound = jnp.percentile(samples, upper_percentile, axis=0)
        
        alpha = (100 - p) / 100.0 * 0.5 + 0.1 # adaptive alpha
        ax.fill_between(x, lower_bound, upper_bound, alpha=alpha, color=color, label=f'{p}% CI')

def plot_predictives(experiment_w: experiment, experiment_m: experiment, intervals=[50, 80, 90]):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 5))
    VGD_color = '#ff7f0e'
    SVGD_color = '#1f77b4'

    plot_shaded_region_predictive(axes[0], experiment_w, experiment_w.particles_SVGD, SVGD_color, intervals)
    plot_shaded_region_predictive(axes[1], experiment_w, experiment_w.particles_VGD, VGD_color, intervals)

    plot_shaded_region_predictive(axes[2], experiment_m, experiment_m.particles_SVGD, SVGD_color, intervals)
    plot_shaded_region_predictive(axes[3], experiment_m, experiment_m.particles_VGD, VGD_color, intervals)