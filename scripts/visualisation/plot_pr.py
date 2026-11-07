from scripts.visualisation.visualisation import *
from scripts.visualisation.utils import draw_zoom_connections

import numpy as np

# ============================================================
# Visualisation Components
# ============================================================

def _compute_eigenvalues(
    G, # Gram matrix
):
    eigs = np.linalg.eigvalsh(G)
    eigs.sort() # sort eigenvalues
    eigs = eigs/len(eigs) # normalize eigenvalues
    eigs = eigs[::-1]

    return eigs

def plot_zoom_lolipops(
    fig,
    ax_bottoms,
    gram_results:dict,
    zoom_layers: list,
    y_vals:dict,
    zoom_model: str,
    plotted_quantity:str,
    fontsize:int,
    lolipop_color=None,
    lolipop_lw = 3,
    lolipop_s=70,
):
    for idx, (layer_idx, ax_sub) in enumerate(zip(zoom_layers, ax_bottoms)):
        layer_name = layer_idx
        # Compute the eigenvalues # TODO
        try: 
            zoomed_eigenvalues = _compute_eigenvalues(gram_results[zoom_model][str(layer_idx)])
        except:
            zoomed_eigenvalues = _compute_eigenvalues(gram_results[zoom_model][layer_idx])
        
        zoomed_cumulative = np.cumsum(zoomed_eigenvalues)
        k = np.arange(len(zoomed_eigenvalues))+1

        # lolipop chart
        ax_sub.vlines(
            k,
            0,
            zoomed_eigenvalues,
            lw=lolipop_lw,
            color=lolipop_color,
            alpha=0.9,
        )
        ax_sub.scatter(
            k,
            zoomed_eigenvalues,
            s=lolipop_s,
            color=lolipop_color,
            edgecolor="white",
            linewidth=lolipop_lw/3,
            zorder=10,
        )

        if idx == 0:
            ax_sub.set_ylabel(
                r"Normalized $\lambda_k$",
                fontsize=0.8 * fontsize,
            )
        else:
            ax_sub.set_yticklabels([])
        if idx == 1:
            ax_sub.set_xlabel(
                "Eigenvalue rank",
                fontsize=0.8 * fontsize,
            )

        # Cumulative
        ax2 = ax_sub.twinx()
        ax2.plot(
            k,
            zoomed_cumulative,
            "--",
            color="grey",
            lw=2*lolipop_lw/3,
            alpha=0.75,
        )
        ax2.set_ylim(0, 1.)
        ax2.tick_params(labelsize=.8*fontsize, colors='grey')

        if idx == len(ax_bottoms)-1:
            ax2.set_ylabel(
                "Cumulative",
                fontsize=0.8 * fontsize,
                color="grey"
            )
        else:
            ax2.set_yticklabels([])

        # Cosmetics
        ax_sub.grid()
        ax_sub.set_title(
            rf"Layer {layer_name} ($PR = {y_vals[layer_idx]:.2f}$)", 
            fontsize=0.8*fontsize, 
            pad=5
        )
        ax_sub.tick_params(labelsize=.8*fontsize)

        ax_sub.set_xlim(min(k)-.5, max(k)+.5)
        ax_sub.set_ylim(0,1)
        ax_sub.set_xticks(k)

        for ax_ in [ax_sub, ax2]:
            ax_.spines["top"].set_visible(False)
            ax_.spines["right"].set_visible(False)

    return

# ============================================================
# MAIN VISUALISATION FIGURE
# ============================================================

def make_zoomed_layers_pr_figure(
    plot_dict: dict,
    results_zoom_data: dict, # Gram matrices
    zoom_model: str,
    zoom_layers: list,
    models_colors: dict,
    plot_main_key: str = "PR",
    plot_ci_key: str = "PR_btstrp",
    scatter_mask_key: str = "p_mask",
    ylabel: str = "PR",
    ylim: tuple = None,
    figsize: tuple = (12, 8),
    fontsize: int = 18,
    lw:int=4,
    ms:int=10,
    lolipop_color=None,
    lolipop_lw = 3,
    lolipop_s=70,
):
    """
    Master function that combines the top layers plot, bottom lolipops, and connectors into one figure.
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

    # 1. Generate top line plot using your existing function
    make_layers_plot(
        plot_dict=plot_dict,
        models_colors=models_colors,
        plot_main_key=plot_main_key,
        plot_ci_key=plot_ci_key,
        scatter_mask_key=scatter_mask_key,
        ax=ax_top,
        figsize=figsize,
        fontsize=fontsize,
        ylabel=ylabel,
        lw=lw,
        ms=ms,
    )

    # Clean up top plot spines/ticks to match draft
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    if not ylim is None:
        ax_top.set_ylim(ylim)

    # 2. Plot bottom lolipops
    zoom_values = {l: plot_dict[zoom_model][plot_main_key][int(l)] for l in zoom_layers}
    # TODO
    plot_zoom_lolipops(
        fig=fig,
        ax_bottoms=ax_bottoms,
        gram_results=results_zoom_data,
        zoom_layers=zoom_layers,
        y_vals=zoom_values,
        plotted_quantity=ylabel,
        zoom_model=zoom_model,
        fontsize=fontsize,
        lolipop_color=models_colors[zoom_model],
        lolipop_lw = lolipop_lw,
        lolipop_s=lolipop_s,
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
    return fig, (ax_top, ax_bottoms)

# ============================================================
# Scores helpers
# ============================================================

def compute_p_value(
    observed_v,
    null_vs,
    smaller:bool=True,
):
    if smaller:
        return (np.sum(null_vs <= observed_v) + 1) / (len(null_vs) + 1)
    else:
        return (np.sum(null_vs >= observed_v) + 1) / (len(null_vs) + 1)

def prepare_pr_for_plot(
    observed_values:dict,
    btstrpd_values:dict,
    null_values:dict=None,
    p_thres:float = .01
):
    models_names = observed_values.keys()

    pr_dict = {
        model_name: {
            "PR": observed_values[model_name],
            "PR_btstrp": [
                (np.percentile(pr, 2.5), np.percentile(pr, 97.5))
                for pr in btstrpd_values[model_name]
            ],
        }
        for model_name in models_names
    }

    if not null_values is None:
        for model_name in models_names:
            pr_dict[model_name]["p_value"] = np.array([
                compute_p_value(
                    observed_v=observed_values[model_name][l], 
                    null_vs=null_values[model_name][l]
                ) for l in np.arange(len(observed_values[model_name]))
            ])
            pr_dict[model_name]["p_mask"] = pr_dict[model_name]["p_value"] <= p_thres

    return pr_dict