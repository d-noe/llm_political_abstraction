from scripts.visualisation.visualisation import *
from scripts.visualisation.utils import bootstrap_ci, draw_zoom_connections

import pandas as pd
from matplotlib.patches import Rectangle

# ============================================================
# Visualisation Components
# ============================================================

def build_heatmap(
    model_layer_results:dict,
):
    unique_domains = sorted(list(set(model_layer_results.keys())))

    df_m_l = pd.DataFrame(index=unique_domains, columns=unique_domains)
    for i in unique_domains:
        for j in unique_domains:
            df_m_l.loc[i,j] = np.mean(
                model_layer_results[i][j]
            )
    
    return df_m_l

def plot_zoom_heatmaps(
    fig,
    ax_bottoms,
    cax,
    raw_results:dict,
    zoom_layers: list,
    y_vals:dict,
    zoom_model: str,
    plotted_quantity:str,
    cmap: str = "PRGn",
    linewidth: float = 0.25,
    fontsize: float = 18,
):
    """
    Plots the detailed breakdown heatmaps for selected layers on the bottom subplots.
    """
    
    for idx, (layer_idx, ax_sub) in enumerate(zip(zoom_layers, ax_bottoms)):
        layer_name = layer_idx

        # Build / retrieve the heatmap data for this layer
        zoomed_heatmap = build_heatmap(raw_results[zoom_model][str(layer_idx)])
        
        # Plot heatmap
        g = sns.heatmap(
            zoomed_heatmap.astype(float),
            ax=ax_sub,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=(idx == len(ax_bottoms) - 1),
            cbar_ax=cax if idx == len(ax_bottoms) - 1 else None,
            linewidths=linewidth,
            linecolor="white",
        )

        ax_sub.set_title(rf"ℓ$={layer_name}$ (${y_vals[layer_idx]:.2f}$)", fontsize=0.8 * fontsize, pad=5)
        ax_sub.tick_params(length=0, labelsize=0.8 * fontsize)
        g.set_xticklabels([t.get_text()[:3] + "." for t in g.get_xticklabels()], rotation=45, ha="center", va="top", fontsize=0.5 * fontsize)

        # Add diagonal hatch patches
        for i in range(len(zoomed_heatmap)):
            ax_sub.add_patch(
                Rectangle(
                    (i, i), 1, 1,
                    fill=False,
                    hatch="///",
                    edgecolor="white",
                    hatch_linewidth=2,
                    alpha=0.5,
                )
            )
            
        if idx > 0:
            ax_sub.set_yticks([])
            ax_sub.set_ylabel("")
        else:
            ax_sub.set_ylabel("Training Domain", fontsize=0.8 * fontsize)
            g.set_yticklabels(
                [t.get_text()[:3] + "." for t in g.get_xticklabels()], 
                fontsize=0.5 * fontsize
            )
        if idx == 1:
            ax_sub.set_xlabel("Evaluation Domain", fontsize=0.8 * fontsize)

    cax.set_ylabel("Cross-domain accuracy", fontsize=0.7 * fontsize)
    cax.tick_params(labelsize=0.7 * fontsize)
    
    return

# ============================================================
# MAIN VISUALISATION FIGURE
# ============================================================

def make_zoomed_layers_figure(
    plot_dict: dict,
    static_baselines_dict: dict,
    results_zoom_data: dict,
    zoom_model: str,
    zoom_layers: list,
    models_colors: dict,
    plot_main_key: str = "CCGP_mean",
    plot_ci_key: str = "CCGP_btstrp",
    ylabel: str = "CCGP",
    ylim: tuple = None,
    figsize: tuple = (12, 8),
    fontsize: int = 18,
    cmap: str = "PRGn",
    lw:int=4,
    heatmaps_lw:float = 0.25,
    ms:int=10,
):
    """
    Function that combines the top layers plot, bottom heatmaps, and connectors into one figure.
    """
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.75, 1],
        width_ratios=[1, 1, 1, 0.05],
        hspace=0.45,
        wspace=0.1,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_bottoms = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cax = fig.add_subplot(gs[1, 3])

    # 1. Generate top line plot using your existing function
    make_layers_plot(
        plot_dict=plot_dict,
        static_baselines_dict=static_baselines_dict,
        models_colors=models_colors,
        plot_main_key=plot_main_key,
        plot_ci_key=plot_ci_key,
        ax=ax_top,
        figsize=figsize,
        fontsize=fontsize,
        ylabel=ylabel,
        lw=lw,
        ms=ms,
    )

    # Clean up top plot spines/ticks to match draft
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), minor=False)
    ax_top.set_yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + 0.5, minor=True)
    if not ylim is None:
        ax_top.set_ylim(ylim)

    # 2. Plot bottom heatmaps
    zoom_values = {l: plot_dict[zoom_model][plot_main_key][int(l)] for l in zoom_layers}
    plot_zoom_heatmaps(
        fig=fig,
        ax_bottoms=ax_bottoms,
        cax=cax,
        raw_results=results_zoom_data,
        zoom_layers=zoom_layers,
        y_vals=zoom_values,
        plotted_quantity=ylabel,
        zoom_model=zoom_model,
        cmap=cmap,
        linewidth=heatmaps_lw,
        fontsize=fontsize,
    )

    # 3. Draw connection patches from top to bottom
    draw_zoom_connections(
        fig=fig,
        ax_top=ax_top,
        ax_bottoms=ax_bottoms,
        zoom_layers=zoom_layers,
        zoom_values=zoom_values,
        zoom_model=zoom_model,
        models_colors=models_colors,
    )

    plt.tight_layout()
    return fig, (ax_top, ax_bottoms, cax)

# ============================================================
# Scores helpers
# ============================================================

def _indiv_layer_pair_scores(
    results_ccgp_model_layer:dict,
    wcp:bool=False,
):
    if wcp:
        return np.array([
            np.mean(results_ccgp_model_layer[k][v])   # average over CV folds
            for k in results_ccgp_model_layer.keys()             # source domain 
            for v in results_ccgp_model_layer[k].keys()          # target domain
            if k == v                                      # within-domain!
        ])
    else:
        return np.array([
            np.mean(results_ccgp_model_layer[k][v])   # average over CV folds
            for k in results_ccgp_model_layer.keys()             # source domain 
            for v in results_ccgp_model_layer[k].keys()          # target domain
            if k != v                                      # cross-domain!
        ])

def _get_pair_scores(
    results_ccgp_model:dict,
    wcp:bool=False,
):
    return {
        l: _indiv_layer_pair_scores(
            results_ccgp_model_layer = results_ccgp_model[str(l)],
            wcp=wcp,
        )
        for l in np.arange(len(results_ccgp_model))
    }

def prepare_classification_scores_for_plot(
    ccgp:bool=True,
    results_ccgp:dict=None,
    loading_args:dict=None,
    results_static:dict=None,
    loading_static_args:dict=None,
):
    assert not ((results_ccgp is None) and (loading_args is None))

    if not loading_args is None:
        results_ccgp = load_res_jsons(**loading_args) # TODO!

    models_names = list(results_ccgp.keys())
    models_layers = {model_name: np.arange(len(results_ccgp[model_name])) for model_name in models_names}

    plot_dict = {model_name: {} for model_name in models_names}
    for model_name in models_names:
        curr_pair_scores = _get_pair_scores(
            results_ccgp[model_name],
            wcp=not ccgp,
        )

        plot_dict[model_name][f"{'CCGP' if ccgp else 'WCP'}_mean"] = np.array([
            np.mean(curr_pair_scores[l])
            for l in models_layers[model_name]
        ])
        plot_dict[model_name][f"{'CCGP' if ccgp else 'WCP'}_btstrp"] = np.array([
            bootstrap_ci(curr_pair_scores[l])
            for l in models_layers[model_name]
        ])

    if not ((results_static is None) and (loading_static_args is None)):
        if not loading_static_args is None:
            results_static = load_res_jsons(**loading_static_args) # TODO!

        static_names = list(results_static.keys())
        
        plot_dict_static = {baseline_name: {} for baseline_name in static_names}
        for baseline_name in static_names:
            base_pair_scores = _indiv_layer_pair_scores(
                results_static[baseline_name], 
                wcp = not ccgp,
            )
            plot_dict_static[baseline_name][f"{'CCGP' if ccgp else 'WCP'}_mean"] = np.mean(base_pair_scores)
            plot_dict_static[baseline_name][f"{'CCGP' if ccgp else 'WCP'}_btstrp"] = bootstrap_ci(base_pair_scores)

        return plot_dict, plot_dict_static
            

    return plot_dict
