import math
import numpy as np
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch

##
MODELS_COLORS = {
    "OLMo-2-1124-7B-Instruct":"hotpink",
    "Llama-3.1-8B":"cornflowerblue",
    "Llama-3.1-8B-Instruct":"royalblue",
    "TF-IDF":"grey",
}
##

sns.set_theme(
    style="ticks",
    context='paper',
)

plt.rcParams.update({
    # "axes.spines.top": False,
    # "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

def make_layers_plot(
    plot_dict:dict,
    static_baselines_dict:dict=None,
    models_colors:dict=MODELS_COLORS,
    plot_main_key:str="mean",
    plot_ci_key:str=None,
    scatter_mask_key:str=None,
    # plotting parameters
    ax=None,
    figsize=(12,8),
    fontsize:int=32,
    lw:int=4,
    ms:int=10,
    # layout & misc.
    ci_aplha:float=.4,
    ylim:tuple=None,
    ylabel:str=None,
):
    """
    Inputs
        - plot_dict [dict] | required
            Dictionary storing the values to be plotted.
            Must be formatted as:
                {
                    [MODEL_NAME_1]: {
                        <plot_main_key>: [list | np.array], # values per layer
                        <plot_ci_key>: [list | np.array], # Opt. CI values per layer
                    },
                    [MODEL_NAME_2]: {...},
                    ...
                }
        - ...
    Outpus
        - fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    models_layers = {model: np.arange(len(plot_dict[model][plot_main_key])) for model in plot_dict.keys()}
    for model in plot_dict.keys():
        # plot line
        p = ax.plot(
            models_layers[model],
            plot_dict[model][plot_main_key],
            color=models_colors[model] if model in models_colors.keys() else None,
            # marker='o', ms=ms,
            lw=lw,
            label=model,
            clip_on=False,
        )
        # plot markers
        if scatter_mask_key is None:
            scatter_mask = np.array([True]*len(models_layers[model]))
        else:
            scatter_mask = plot_dict[model][scatter_mask_key]
        ax.scatter(
            models_layers[model][scatter_mask],
            np.array(plot_dict[model][plot_main_key][scatter_mask]),
            marker='o',
            s=10*ms,
            color=models_colors[model] if model in models_colors.keys() else None,
            clip_on=False,
        )
        # plot CI
        if not plot_ci_key is None:
            curr_model_cis = plot_dict[model][plot_ci_key]
            ax.fill_between(
                x = models_layers[model],
                y1 = [ci[0] for ci in curr_model_cis],
                y2 = [ci[1] for ci in curr_model_cis],
                color = p[0].get_color(),
                alpha= ci_aplha,
            )

    # Plot 'static' baselines
    if not static_baselines_dict is None:
        xmin, xmax = 0, np.max([len(l)-1 for l in models_layers.values()])#ax.get_xlim()

        for k, v in static_baselines_dict.items():
            q = ax.axhline(
                y = v[plot_main_key],
                xmin=xmin, xmax=xmax,
                color=models_colors[k] if k in models_colors.keys() else None,
                ls='--',
                lw=lw,
                label=k
            )
            if (not plot_ci_key is None) and (plot_ci_key in v.keys()):
                ax.fill_between(
                    x = np.linspace(xmin, xmax, 2),
                    y1 = [v[plot_ci_key][0]]*2,
                    y2 = [v[plot_ci_key][1]]*2,
                    color=q.get_color(),
                    alpha=ci_aplha,
                )

    ###
    ax.tick_params(axis='both', which='major', labelsize=0.8*fontsize)

    xlim_plot_max = np.max([len(l)-1 for l in models_layers.values()])
    ax.set_ylim(ylim)
    ax.set_xlim([0,xlim_plot_max])

    ax.legend(fontsize=0.8*fontsize)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel("Layer", fontsize=fontsize)
            
    return fig, ax


def make_detailed_plots(
    layers_heatmaps:dict,
    #
    main_title:str="Detailed Results",
    represented_value:str="Cross-Condition Performance",
    xlabel:str="Evaluation Domain",
    ylabel:str="Training Domain",
    #
    cmap="PRGn",
    n_cols=5,
    individual_plot_size=3.4,
    linewidth=0.25,
    fontsize=24,
    vmin=0,
    vmax=1,
    triangular=False
    # 
):  
    # --------------
    # plot detailed results
    layer_keys = sorted(layers_heatmaps.keys(), key=lambda x: int(x))
    n_layers = len(layer_keys)
    n_rows = math.ceil(n_layers / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(individual_plot_size * n_cols,
                    individual_plot_size * n_rows + 1.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_2d(axes)

    # -------------
    # Plot Heatmaps
    for idx, l in enumerate(layer_keys):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        sns.heatmap(
            layers_heatmaps[l].astype(float),
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
            linewidths=linewidth,
            linecolor="white",
            mask=np.triu(layers_heatmaps[l], k=1) if triangular else np.zeros(layers_heatmaps[l].shape)
        )

        ax.set_title(l, fontsize=0.6*fontsize, pad=5)
        ax.tick_params(length=0, labelsize=.6*fontsize)

        for i in range(len(layers_heatmaps[l])): # Mark on diagonal values (within-domain)
            ax.add_patch(
                Rectangle(
                    (i, i),
                    1,
                    1,
                    fill=False,
                    hatch="///",
                    edgecolor="white",
                    hatch_linewidth=2,
                    alpha=.5,
                )
            )

    # -----------------
    # Remove unused axes
    for idx in range(n_layers, n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    # --------------------
    # Make shared colorbar
    sm = ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        location='top',
        # fraction=0.01,
        pad=0.005,
        aspect=45,
    )
    cbar.ax.tick_params(labelsize=.6*fontsize)
    cbar.set_label(
        represented_value,
        fontsize=0.8*fontsize,
        labelpad=8,
    )

    # ---------------------
    # Title & shared labels
    fig.suptitle(
        main_title,
        fontsize=fontsize,
        fontweight="bold",
    )

    fig.supxlabel(
        xlabel,
        fontsize=.8*fontsize,
        fontweight="semibold",
    )
    fig.supylabel(
        ylabel,
        fontsize=.8*fontsize,
        fontweight="semibold",
    )

    return fig, axes


def make_correlation_plot(
    plot_dict:dict, # {model_name: {x_value: [], y_value: []}, ...}
    models_colors:dict,
    x_value:str="CCGP",
    y_value:str="PR",
    figsize=(7,7),
    linewidth=2,
    markersize=5,
    fontsize=22,
):
    fig, ax = plt.subplots(figsize=figsize,)

    models_names = list(plot_dict.keys())

    for m in models_names:
        x = plot_dict[m][x_value]
        y = plot_dict[m][y_value]

        corr = pearsonr(x, y)

        # Draw the complete trajectory
        ax.plot(
            x,
            y,
            c=models_colors[m],
            # label=rf"{m}"+"\n"+rf"($r$ = {pr.statistic:.2f})",
            label=rf"{m} ($r$ = {corr.statistic:.2f})",
            marker='o',
            alpha=.8,
            lw=linewidth,
            ms=markersize,
        )

        # Add an arrowhead in the middle of every segment
        for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # midpoint
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            # small vector in the direction of travel
            dx = (x1 - x0) * 0.001
            dy = (y1 - y0) * 0.001

            arrow = FancyArrowPatch(
                (xm - dx, ym - dy),
                (xm + dx, ym + dy),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=.75*linewidth,
                edgecolor=models_colors[m],
                facecolor="none",       # unfilled arrowhead
            )

            ax.add_patch(arrow)

    # cosmetics
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=0.8*fontsize
    )
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.3),
        fontsize=0.8*fontsize
    )
    ax.set_xlabel(x_value, fontsize=fontsize)
    ax.set_ylabel(y_value, fontsize=fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax

# ============================================================
# MAIN
# ============================================================

def parse_args():
    import argparse
    import os 
    os.sys.path.append('./')
    from scripts.utils import load_config_dict

    # 1. First, parse a preliminary parser just to find the config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--exp_config", 
        type=str, 
        default="./configs/exp_config.json",
        help="Path to the JSON configuration file."
    )
    pre_args, unknown  = pre_parser.parse_known_args()

    # 2. Load the JSON config file if it exists
    file_config = load_config_dict(
        pre_args.exp_config, 
        verbose=False
    )

    # 3. Set up the main parser, using JSON values as defaults if available
    parser = argparse.ArgumentParser(
        description="Output Visualisations."
    )

    parser.add_argument(
        "--exp_config", 
        type=str, 
        default=pre_args.exp_config
    )

    parser.add_argument(
        "--ccgp",
        action="store_true",
        default=file_config.get("ccgp_viz", False),
        help="Make Cross-Condition Generalisation Performance (CCGP) visualisations.",
    )
    parser.add_argument(
        "--include_tfidf", "-tfidf",
        action="store_true",
        default=file_config.get("compute_tfidf", False),
        help="Include TFIDF lexical baseline.",
    )
    parser.add_argument(
        "--pr",
        action="store_true",
        default=file_config.get("pr_viz", False),
        help="Make Participation Ratio (PR) visualisations.",
    )
    parser.add_argument(
        "--corr",
        action="store_true",
        default=file_config.get("corr_viz", False),
        help="Make CCGP/PR correlation plot.",
    )

    parser.add_argument(
        "--pr_B",
        type=int,
        default=file_config.get("bootstrap_samples", 100),
        help="Number of bootstraped used in PR computation (files stored as 'bootstraped_[pr_B]_prs_<MODEL_NAME>.npy)"
    )
    parser.add_argument(
        "--pr_perms",
        type=int,
        default=file_config.get("n_permutations", 200),
        help="Number of permutations used in PR null computation (files stored as 'null_[pr_perms]_prs_<MODEL_NAME>.npy)"
    )
    parser.add_argument(
        "--figures_format", "-f",
        type=str,
        default=file_config.get("figures_format", "pdf"),
        help="Format of the saved figures.",
    )
    parser.add_argument(
        "--zoom_model", "-m",
        type=str,
        default=file_config.get("zoom_model", "Llama-3.1-8B-Instruct"),
        help="Models for zoomed in detailed figures.",
    )
    parser.add_argument(
        "--zoom_layers", "-l",
        nargs="+",
        default=file_config.get("zoom_layers", [2,14,30]),
        help="Layers for zoomed in detailed figures (must be exactly 3 layers!).",
    )
    parser.add_argument(
        "--raw_outputs_folder_path", "-i",
        type=str,
        default=file_config.get("raw_outputs_folder", "./outputs/raw_outputs/"),
        help="Base folder path containing the raw outputs of the analysis (with subfolders 'ccgp' and 'pr') (default: ./outputs/raw_outputs/).",
    )
    parser.add_argument(
        "--visualisation_output_folder", "-o",
        type=str,
        default=file_config.get("visualisation_output_folder", "./outputs/figures/"),
        help="Base folder path to save the results (default: ./outputs/raw_outputs/ccgp/).",
    )
    parser.add_argument(
        "--dataset_name", 
        type=str,
        default=file_config.get("dataset_name", "twinviews"),
        help="Name of the dataset."
    )
    parser.add_argument(
        "--models_names", 
        nargs="+",
        default=file_config.get("models_names", ["Llama-3.1-8B","Llama-3.1-8B-Instruct","OLMo-2-1124-7B-Instruct"]),
        help="Names of the models to use."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=file_config.get("verbose", False),
        help="Enable verbose output."
    )

    return parser.parse_args()


if __name__ == "__main__":
    import os
    os.sys.path.append("./")
    import argparse

    from scripts.visualisation.plot_classification import *
    from scripts.visualisation.plot_pr import *
    from scripts.visualisation.utils import bootstrap_ci, draw_zoom_connections
    from scripts.visualisation.utils import draw_zoom_connections
    from scripts.utils import *

    # 0. Parse arguments
    args = parse_args()

    verbose = args.verbose
    save_format = args.figures_format
    if save_format == 'png':
        save_fig_args = {
            'dpi':300,
            'transparent':True,
        }
    else:
        save_fig_args = {}

    base_path = args.raw_outputs_folder_path
    output_folder = args.visualisation_output_folder
    dataset_name = args.dataset_name
    models_names = args.models_names

    zoom_model = args.zoom_model
    assert (zoom_model in models_names), f"'zoom_model' must be one of the 'models_names': {models_names}."
    zoom_layers = args.zoom_layers
    zoom_layers = [int(l) for l in zoom_layers]

    B = args.pr_B
    n_permutations = args.pr_perms

    do_ccgp = args.ccgp
    do_include_tfidf = args.include_tfidf
    do_pr = args.pr
    do_corr = args.corr

    # 1. CCGP Figures
    if do_ccgp:
        raw_ccgp_folder = os.path.join(base_path, 'ccgp')
        ccgp_output_folder = os.path.join(output_folder, dataset_name, "ccgp")
        os.makedirs(ccgp_output_folder, exist_ok=True)
        if verbose:
            print("Preparing Cross-Condition Generalisation Performance (CCGP) Figures.")
        
        # 1.1 Loading raw results
        if verbose:
            print("... loading raw results.")
        models_layers = {}
        results_ccgp_label = {}

        for model_name in models_names:
            file_ccgp_path = os.path.join(raw_ccgp_folder, dataset_name, f"{model_name}.json")

            results_ccgp_label[model_name] = load_json_dict(file_ccgp_path)
            models_layers[model_name] = np.arange(len(results_ccgp_label[model_name]))

        if do_include_tfidf:
            tfidf_file_ccgp_path = os.path.join(raw_ccgp_folder, dataset_name, f"TFIDF.json")
            baselines_results_ccgp_label = {"TF-IDF": load_json_dict(tfidf_file_ccgp_path)}
        else:
            baselines_results_ccgp_label = None

        # 1.2 Within-Condition Performance (WCP)
        if verbose:
            print("... making Within-Condition Performance (WCP) figure.")
        wcp_fig_filepath = os.path.join(
            ccgp_output_folder,
            f"wcp_btstrp_zoomed.{save_format}"
        )
        wcp_models, wcp_tfidf  = prepare_classification_scores_for_plot(
            ccgp=False, # -> WCP
            results_ccgp=results_ccgp_label,
            results_static=baselines_results_ccgp_label,
            loading_static_args=None,
        )
        fig, ax = make_layers_plot(
            plot_dict=wcp_models,
            static_baselines_dict=wcp_tfidf,
            plot_main_key="WCP_mean",
            plot_ci_key="WCP_btstrp",
            # ylim=(0.94,1),
            ylabel="WCP",
            models_colors=MODELS_COLORS,
            figsize=(8,4),
            fontsize=32,
            lw=6,
            ms=15,
        )
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(
            wcp_fig_filepath,
            bbox_inches='tight',
            **save_fig_args
        )
        if verbose:
            print(f"\t... WCP fig saved at: {wcp_fig_filepath}.")

        # 1.3 Cross-Condition Generalisation Performance (CCGP)
        if verbose:
            print("... making Cross-Condition Generalisation Performance (CCGP) figure.")
        ccgp_fig_filepath = os.path.join(
            ccgp_output_folder,
            f"ccgp_btstrp_zoomed_{zoom_model}_{'_'.join(np.array(zoom_layers).astype(str))}.{save_format}"
        )
        ccgp_models, ccgp_tfidf  = prepare_classification_scores_for_plot(
            ccgp=True, # -> CCGP
            results_ccgp=results_ccgp_label,
            results_static=baselines_results_ccgp_label,
            loading_static_args=None,
        )
        fig, axes = make_zoomed_layers_figure(
            plot_dict=ccgp_models,
            static_baselines_dict=ccgp_tfidf,
            results_zoom_data=results_ccgp_label,  # raw data dict for building heatmaps
            zoom_model=zoom_model,
            zoom_layers=zoom_layers,
            models_colors=MODELS_COLORS,
            plot_main_key="CCGP_mean",
            plot_ci_key="CCGP_btstrp",
            ylabel="CCGP",
            ylim=(0.5,1),
            figsize=(10, 10),
            cmap="PRGn",
            # cmap="BrBG_r",

            fontsize=32,
            lw=6,
            ms=15,
        )
        plt.savefig(
            ccgp_fig_filepath,
            bbox_inches='tight',
            **save_fig_args
        )
        if verbose:
            print(f"\t... CCGP fig saved at: {ccgp_fig_filepath}.")

        # 1.4 Detailed Cross-condition Performance Plots
        if verbose:
            print("... making detailed cross-domain performance figures.")
        models_layers_heatmaps = {
            m: {
                l: build_heatmap(
                    results_ccgp_label[m][str(l)]
                )
                for l in models_layers[m]
            }
            for m in models_names
        }
        for m in models_names:
            detailed_cp_m_fig_filepath = os.path.join(
                ccgp_output_folder,
                f"detailed_{m}.{save_format}"
            )
            make_detailed_plots(
                models_layers_heatmaps[m],
                main_title=m,
            )
            plt.savefig(
                detailed_cp_m_fig_filepath,
                bbox_inches='tight',
                **save_fig_args
            )
            if verbose:
                print(f"\t... detailed performance for {m} saved at {detailed_cp_m_fig_filepath}.")

    # 2. PR Figures
    if do_pr:
        raw_pr_folder = os.path.join(base_path, 'pr')
        pr_output_folder = os.path.join(output_folder, dataset_name, "pr")
        os.makedirs(pr_output_folder, exist_ok=True)
        if verbose:
            print("Preparing Participation Ratio (PR) Figures.")
        # 2.1 Loading raw results
        if verbose:
            print("... loading raw results.")
        # Observed
        observed_prs = {}
        observed_grams = {}
        models_layers = {}
        for model_name in models_names:
            file_prs_obs_path = os.path.join(raw_pr_folder, dataset_name, f"observed_prs_{model_name}.npy")
            file_grams_obs_path = os.path.join(raw_pr_folder, dataset_name, f"observed_grams_{model_name}.npy")

            observed_prs[model_name] = load_array_npy(file_prs_obs_path)
            observed_grams[model_name] = load_array_npy(file_grams_obs_path)

            models_layers[model_name] = np.arange(len(observed_prs[model_name]))
        # Bootstraped
        models_bootstrap_prs = {}
        models_bootstrap_grams = {}
        for model_name in models_names:
            file_prs_btstrp_path = os.path.join(raw_pr_folder, dataset_name, f"bootstraped_{B}_prs_{model_name}.npy")
            file_grams_btstrp_path = os.path.join(raw_pr_folder, dataset_name, f"bootstraped_{B}_grams_{model_name}.npy")

            models_bootstrap_prs[model_name] = load_array_npy(file_prs_btstrp_path)
            models_bootstrap_grams[model_name] = load_array_npy(file_grams_btstrp_path)
        # Null
        models_null_prs = {}
        for model_name in models_names:
            file_prs_null_path = os.path.join(raw_pr_folder, dataset_name, f"null_{n_permutations}_prs_{model_name}.npy")

            models_null_prs[model_name] = load_array_npy(file_prs_null_path)

        pr_plot_dict = prepare_pr_for_plot(
            observed_prs,
            models_bootstrap_prs,
            models_null_prs,
        )

        pr_fig_filepath = os.path.join(
            pr_output_folder,
            f"pr_btstrp_zoomed_{zoom_model}_{'_'.join(np.array(zoom_layers).astype(str))}.{save_format}"
        )
        fig, axes = make_zoomed_layers_pr_figure(
            plot_dict=pr_plot_dict,
            results_zoom_data=observed_grams,
            zoom_model=zoom_model,
            models_colors=MODELS_COLORS,
            zoom_layers=zoom_layers
        )
        plt.savefig(
            pr_fig_filepath,
            bbox_inches='tight',
            **save_fig_args
        )
        if verbose:
            print(f"\t... PR fig saved at: {pr_fig_filepath}.")

        # 2.2 Detailed Results
        if verbose:
            print("... making detailed cross-domain performance figures.")

        models_layers_heatmaps_grams = {
            m: {
                l: observed_grams[m][l]
                for l in models_layers[m]
            }
            for m in models_names
        }
        for m in models_names:
            detailed_sim_m_fig_filepath = os.path.join(
                pr_output_folder,
                f"detailed_{m}.{save_format}"
            )
            make_detailed_plots(
                models_layers_heatmaps_grams[m],
                main_title=m,
                xlabel=r"Domain $d_j$",
                ylabel= r"Domain $d_i$",
                cmap="managua_r",
                vmin=-1, vmax=1,
                represented_value="Contrast Vectors Cosine Similarity",
                triangular=True
            )
            plt.savefig(
                detailed_sim_m_fig_filepath,
                bbox_inches='tight',
                **save_fig_args
            )
            if verbose:
                print(f"\t... detailed contrast vectors similarities for {m} saved at {detailed_sim_m_fig_filepath}.")



    # 3. Correlation plot
    if do_corr:
        if verbose:
            print("Preparing CCGP/PR correlation Figure.")
        # 3.1 (re)Loading raw results
        # CCGP
        raw_ccgp_folder = os.path.join(base_path, 'ccgp')
        results_ccgp_label = {}
        for model_name in models_names:
            file_ccgp_path = os.path.join(raw_ccgp_folder, dataset_name, f"{model_name}.json")
            results_ccgp_label[model_name] = load_json_dict(file_ccgp_path)
        ccgp_models  = prepare_classification_scores_for_plot(
            ccgp=True,
            results_ccgp=results_ccgp_label,
        )
        # PR
        raw_pr_folder = os.path.join(base_path, 'pr')
        results_pr = {}
        for model_name in models_names:
            file_prs_obs_path = os.path.join(raw_pr_folder, dataset_name, f"observed_prs_{model_name}.npy")
            results_pr[model_name] = load_array_npy(file_prs_obs_path)
        # Combine
        plot_dict = {
            m: {
                "CCGP": ccgp_models[m]['CCGP_mean'],
                "PR": results_pr[m]
            }
            for m in models_names
        }

        corr_fig_filepath = os.path.join(
            output_folder, dataset_name,
            f"ccgp_pr_correlation_plot.{save_format}"
        )
        make_correlation_plot(
            plot_dict=plot_dict, # {model_name: {x_value: [], y_value: []}, ...}
            models_colors=MODELS_COLORS
        )
        plt.savefig(
            corr_fig_filepath,
            bbox_inches='tight',
            **save_fig_args
        )
        if verbose:
            print(f"\t... CCGP/PR corr plot saved at {corr_fig_filepath}.")

