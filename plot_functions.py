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

fontsize_axis = 20

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
        label=f'True MMD: {actual_mmd:.3e}' # 添加图例标签
    )
    sci_formatter = ticker.FormatStrFormatter('%.0e')
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.tick_params(axis='both', labelsize=fontsize_axis)
    # ax.legend()
    # return all_mmd_values, actual_mmd


def plot_diagnostic_manual_broken(
    ax_left, ax_right, 

    all_mmd_values=None, actual_mmd=None,
    # 传入您在 gridspec_kw 中定义的宽度比，用于绘制平行的 //
    width_ratios=[0.9, 0.3]
):
    
        
    print(f"Actual mmd", actual_mmd)

    # --- 2. JAX -> NumPy 转换 ---
    all_mmd_values_np = np.array(all_mmd_values)
    actual_mmd_float = float(actual_mmd)
    total_count = len(all_mmd_values_np)
    weights = np.ones_like(all_mmd_values_np) / total_count

    # --- 3. 动态计算 xlims ---
    hist_min = np.min(all_mmd_values_np)
    hist_max = np.max(all_mmd_values_np)
    hist_pad = (hist_max - hist_min) * 0.1 # 10% padding
    xlim1 = (hist_min - hist_pad, hist_max + hist_pad)
    
    xlim2_pad = (hist_max - hist_min) * 0.1 
    xlim2 = (actual_mmd_float - xlim2_pad, actual_mmd_float + xlim2_pad)

    # --- 4. 在两个子图上都画上所有元素 (核心) ---
    
    # 在 ax_left 和 ax_right 上都画
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
    
    
    # --- 5. 设置各自的 X 轴范围 (实现“断轴”) ---
    ax_left.set_xlim(xlim1)
    ax_right.set_xlim(xlim2)

    # --- 5.5 【关键新增】手动同步 Y 轴范围 ---
    y_lims = ax_left.get_ylim()
    ax_right.set_ylim(y_lims)

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
    # ax_left.set_xticks(...) # (可选)

    sci_formatter = FormatStrFormatter('%.1e')
    ax_left.xaxis.set_major_formatter(sci_formatter)
    

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


def plot_main_figure(file_name='main_fig.dill', ):
    sns.set_theme(
    style="white",
    rc={
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "legend.frameon": True
    }
    )
    # -----------------------------------------------------------

    jax.config.update("jax_enable_x64", True)
    try:
        with open('main_fig.dill', 'rb') as f:
            data = dill.load(f)
        print(f"成功加载文件，包含 {len(data.keys())} 个数组: {list(data.keys())}")
    except FileNotFoundError:
        print(r"错误：'main_fig.dill' 文件未找到。")
        exit()

    # --- 3. 自动化绘图 ---
    col_widths = [1, 1, 1, 1, 1, 0.75 + 0.25] 
    row_heights = [1, 1, 1]
    nested_col_widths = [0.75, 0.25]

    # --- 创建图像 ---
    fig = plt.figure(figsize=(24, 12.5))
    main_gs = fig.add_gridspec(
        nrows=3, 
        ncols=6, 
        height_ratios=row_heights,
        width_ratios=col_widths
    )
    axes = np.empty((3, 7), dtype=object)
    for r in range(3):
        # --- a. 正常添加前 5 列 (索引 0-4) ---
        for c in range(5):
            axes[r, c] = fig.add_subplot(main_gs[r, c])
            
        # --- b. 添加第 6 列 (索引 5) 作为容器 ---
        # 获取主网格的第 (r, 5) 个单元格
        sub_gs_spec = main_gs[r, 5]
        
        # 在该单元格内创建一个 1x2 的嵌套子网格
        # 【【【 这就是您要的控制！wspace=0.05 】】】
        nested_gs = sub_gs_spec.subgridspec(
            1, 2, 
            width_ratios=nested_col_widths, 
            wspace=0.05 # <--- 在此独立控制最后两列的间距
        )
        
        # --- c. 在子网格中添加最后两列 (索引 5 和 6) ---
        axes[r, 5] = fig.add_subplot(nested_gs[0, 0])
        axes[r, 6] = fig.add_subplot(nested_gs[0, 1])

        
    data_prefixes = ['quad', 'sig', 'landq']
    # row_titles = ['Quadratic', 'Sigmoid', '2D Quadratic']
    row_titles = [r'$\mathrm{Quadratic}$', r'$\mathrm{Sigmoid}$', r'$\mathrm{Linear (d=2)}$']
    VGD_color = '#ff7f0e'
    SVGD_color = '#1f77b4'

    for row_idx, prefix in enumerate(data_prefixes):
        experiment_w = data[f'experiment_{prefix}_w']
        scatter_data_w = experiment_w.data
        xg = np.sort(scatter_data_w[0])
        particles_SVGD_w = experiment_w.particles_SVGD
        particles_VGD_w = experiment_w.particles_VGD
        all_mmd_values_w = data[f'all_mmd_values_{prefix}_w']
        actual_mmd_w = data[f'actual_mmd_{prefix}_w'].item()

        plot_shaded_region_predictive(axes[row_idx, 0], experiment_w, experiment_w.particles_SVGD, SVGD_color)
        plot_shaded_region_predictive(axes[row_idx, 1], experiment_w, experiment_w.particles_VGD, VGD_color)

        ax = axes[row_idx, 2]
        plot_diagnostic(ax, all_mmd_values=all_mmd_values_w, actual_mmd=actual_mmd_w)

        experiment_m = data[f'experiment_{prefix}_m']
        scatter_data_m = experiment_m.data
        xg = np.sort(scatter_data_m[0])
        particles_SVGD_m = experiment_m.particles_SVGD
        particles_VGD_m = experiment_m.particles_VGD
        all_mmd_values_m = data[f'all_mmd_values_{prefix}_m']
        actual_mmd_m = data[f'actual_mmd_{prefix}_m'].item()

        plot_shaded_region_predictive(axes[row_idx, 3], experiment_m, particles_SVGD_m, SVGD_color)
        plot_shaded_region_predictive(axes[row_idx, 4], experiment_m, particles_VGD_m, VGD_color)

        ax_left = axes[row_idx, 5]
        ax_right = axes[row_idx, 6]
        ax_right.sharey(ax_left)
        plot_diagnostic_manual_broken(
            ax_left, 
            ax_right, 
            all_mmd_values=all_mmd_values_m, 
            actual_mmd=actual_mmd_m,
            width_ratios=[0.9, 0.3] 
        )

    # --- 4. 调整布局 ---
    plt.tight_layout(rect=[0.05, 0, 1, 0.93])

    # --- 5. 添加全局标题和标签 ---
    col_titles = [r'$P_\mathrm{Bayes}$', r'$P_\mathrm{PrO}$', r'$\mathrm{MMD}$', r'$P_\mathrm{Bayes}$', r'$P_\mathrm{PrO}$', r'$\mathrm{MMD}$']
    for i, title in enumerate(col_titles):
        if title:
            if i == 5: # (索引为5的标题)
                # (特殊处理：跨越第 5 和 第 6 列)
                x0 = axes[0, 5].get_position().x0
                x1 = axes[0, 6].get_position().x1
                x_coord = (x0 + x1) / 2
            else:
                x_coord = (axes[0, i].get_position().x0 + axes[0, i].get_position().x1) / 2
            
            fig.text(x_coord, 0.96, title, ha='center', va='top', fontsize=28)

    for i, title in enumerate(row_titles):
        y_coord = (axes[i, 0].get_position().y0 + axes[i, 0].get_position().y1) / 2
        fig.text(0.02, y_coord, title, ha='left', va='center', fontsize=28, rotation=90)


    # --- 6. 显示图像 ---
    fig.subplots_adjust(
        wspace=0.2,   # 水平间距
        hspace=0.15    # 垂直间距
    )
    plt.show()