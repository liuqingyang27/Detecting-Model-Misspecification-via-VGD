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

import dill

fontsize_axis = 22
VGD_colour = '#ff7f0e'
SVGD_colour = '#1f77b4'
def plot_KGD(ax, experiment: experiment):
    ax.plot(range(len(experiment.history_KGD)), jnp.log(experiment.history_KGD), label='Log KGD', color=VGD_colour)
    ax.plot(range(len(experiment.history_KSD)), jnp.log(experiment.history_KSD), label='Log KSD', color=SVGD_colour)
    ax.set(xlabel=None, ylabel=None)


def plot_kde_q(ax, experiment: experiment):
    sns.kdeplot(
        jnp.array(experiment.particles_SVGD), 
        fill=False, 
        label='Q_Bayes', 
        clip=(0, None), 
        ax=ax, 
        color=SVGD_colour, 
        alpha=0.6
    )
    sns.kdeplot(
        jnp.array(experiment.particles_VGD), 
        fill=False, 
        label='Q_PrO', 
        clip=(0, None), 
        ax=ax, 
        color=VGD_colour, 
        alpha=0.6
    )
    ax.set(xlabel=None, ylabel=None)
    # ax.legend()

def plot_kde_q_2d(ax, experiment: experiment, xlims=[-4,4], ylims=[-4,4]):
    x = experiment.particles_SVGD[:, 0]
    y = experiment.particles_SVGD[:, 1]
    sns.kdeplot(
        x=x, 
        y=y, 
        thresh=0.05, 
        levels=100, 
        ax=ax, 
        color=SVGD_colour,
        alpha=0.6,
        fill=False
    )
    x = experiment.particles_VGD[:, 0]
    y = experiment.particles_VGD[:, 1]
    sns.kdeplot(
        x=x, 
        y=y, 
        thresh=0.05, 
        levels=100, 
        ax=ax, 
        color=VGD_colour,
        alpha=0.6,
        fill=False
    )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    handles = [Line2D([0], [0], color=SVGD_colour, lw=2),
                   Line2D([0], [0], color=VGD_colour, lw=2)]
    labels = [r'$Q_{Bayes}$', r'$Q_{PC}$']
    ax.set(xlabel=None, ylabel=None)
    # ax.legend()

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
    ax.tick_params(axis='both', labelsize=fontsize_axis)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))

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

def plot_diagnostic(ax, all_mmd_values=None, actual_mmd=None):
    print("Actual mmd", actual_mmd)

    all_mmd_values_np = np.array(all_mmd_values)
    total_count = len(all_mmd_values_np)
    # weights = np.ones_like(all_mmd_values_np) / total_count

    sns.kdeplot(all_mmd_values, fill=True, label='MMD (KDE)', clip=(0, None), ax=ax)
    # ax.set_ylabel('Probability (Count / Total)')
    ax.set_ylabel(None)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # ax.hist(all_mmd_values_np, bins=30, density=False, alpha=0.6, weights=weights)
    ax.axvline(
        x=actual_mmd, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'True MMD: {actual_mmd:.3e}'
    )
    sci_formatter = ticker.FormatStrFormatter('%.0e')
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.tick_params(axis='both', labelsize=fontsize_axis)
    # ax.legend()
    # return all_mmd_values, actual_mmd


def plot_diagnostic_manual_broken(
    ax_left, ax_right, 

    all_mmd_values=None, actual_mmd=None,
    width_ratios=[0.9, 0.3],
    xlim_left=None,
    xticks_left=None
):
    
        
    print(f"Actual mmd", actual_mmd)

    all_mmd_values_np = np.array(all_mmd_values)
    actual_mmd_float = float(actual_mmd)
    total_count = len(all_mmd_values_np)
    weights = np.ones_like(all_mmd_values_np) / total_count

    hist_min = np.min(all_mmd_values_np)
    hist_max = np.max(all_mmd_values_np)
    hist_pad = (hist_max - hist_min) * 0.1 
    xlim1 = (hist_min - hist_pad, hist_max + hist_pad)
    if xlim_left is not None:
        xlim1 = xlim_left
    
    xlim2_pad = (hist_max - hist_min) * 0.1 
    xlim2 = (actual_mmd_float - xlim2_pad, actual_mmd_float + xlim2_pad)

    
    for ax in [ax_left, ax_right]:
        # ax.hist(all_mmd_values_np, bins=30, density=False, alpha=0.6, weights=weights)
        sns.kdeplot(all_mmd_values, fill=True, label='MMD (KDE)', clip=(0, None), ax=ax)
        ax.axvline(
            x=actual_mmd_float, 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'True MMD: {actual_mmd_float:.4f}'
        )
    
    
    ax_left.set_xlim(xlim1)

    # --- 关键修改开始 ---
    if xlim_left is not None:
        # 情况 A：指定了特定范围，强制显示这两个数
        ax_left.set_xticks(xticks_left)
        
        # 这里的 formatter 确保它们以科学计数法显示
        sci_formatter = FormatStrFormatter('%.1e')
        ax_left.xaxis.set_major_formatter(sci_formatter)
        
        # 【重要】绝对不要在这里再次调用 MaxNLocator，否则上面的 set_xticks 会失效
        
    else:
        # 情况 B：没有指定范围，使用之前的自动修剪逻辑
        ax_left.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune='both'))
        
        sci_formatter = FormatStrFormatter('%.1e')
        ax_left.xaxis.set_major_formatter(sci_formatter)
    # --- 关键修改结束 ---


    ax_right.set_xlim(xlim2)
    y_lims = ax_left.get_ylim()
    ax_right.set_ylim(y_lims)
    
    # ... 后续隐藏边框的代码保持不变 ...

    # --- 6. 隐藏和调整 Spines 和 Ticks ---
    ax_left.spines['right'].set_visible(False)  # 隐藏左图的右边框
    ax_right.spines['left'].set_visible(False)   # 隐藏右图的左边框
    ax_left.yaxis.set_major_formatter(ticker.NullFormatter()) # 隐藏左图的 y 轴刻度标签
    ax_right.yaxis.set_major_formatter(ticker.NullFormatter()) # 隐藏右图的 y 轴刻度标签

    # 隐藏右图的 y 轴刻度线和标签
    ax_right.yaxis.set_ticks_position('none')
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # 调整刻度
    ax_right.set_xticks([actual_mmd_float]) 
    ax_right.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1e'))
 
    

    # --- 7. 绘制【已修正】的平行断轴斜线 ( // ) ---
    d_y = .015 # Y 轴半高
    d_x_base = .015 # X 轴基准半宽 (对应比例最小的图)
    min_ratio = min(width_ratios)

    # ax_left (左图, 比例 0.6)
    d_x1 = d_x_base * (min_ratio / width_ratios[0]) # dx * (0.3 / 0.6)
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
    ax_left.plot((1 - d_x1, 1 + d_x1), (-d_y, +d_y), **kwargs) # 右下
    ax_left.plot((1 - d_x1, 1 + d_x1), (1 - d_y, 1 + d_y), **kwargs) # 右上

    # ax_right (右图, 比例 0.3)
    d_x2 = d_x_base * (min_ratio / width_ratios[1]) # dx * (0.3 / 0.3)
    kwargs.update(transform=ax_right.transAxes) 
    ax_right.plot((-d_x2, +d_x2), (-d_y, +d_y), **kwargs) # 左下
    ax_right.plot((-d_x2, +d_x2), (1 - d_y, 1 + d_y), **kwargs) # 左上

    # # --- 8. 添加图例和标签 ---
    # ax_right.legend() 
    # ax_left.set_ylabel('Probability (Count / Total)') # <--- 更新标签
    ax_left.set_ylabel(None)
    ax_right.set_ylabel(None)
    ax_left.tick_params(axis='both', labelsize=fontsize_axis)
    ax_right.tick_params(axis='both', labelsize=fontsize_axis)
    # 统一设置 X 轴标签 (在 figure 级别设置更佳，但在这里也行)
    # 我们可以把标签放在两个子图的中间
    # fig = ax_left.get_figure()
    # x_center = (ax_left.get_position().x1 + ax_right.get_position().x0) / 2
    # y_pos = ax_left.get_position().y0 - 0.1 # (需要微调)
    # fig.text(x_center, y_pos, 'MMD Value', ha='center', va='top')


