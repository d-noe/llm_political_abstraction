import numpy as np

from matplotlib.patches import ConnectionPatch


def bootstrap_ci(values, percentiles=(2.5, 97.5), n_btstrp=1000):
    n_data = len(values)
    values_means = []
    for _ in range(n_btstrp):
        values_means += [
            np.random.choice(
                values,
                size=n_data,
                replace=True
            ).mean()
        ]

    return np.percentile(values_means, percentiles[0]), np.percentile(values_means, percentiles[1])


def draw_zoom_connections(
    fig,
    ax_top,
    ax_bottoms,
    zoom_layers: list,
    zoom_values: dict,
    zoom_model: str,
    models_colors: dict,
):
    """
    Highlights points on the top plot and draws dashed connection patches to the bottom subplots.
    """
    model_color = models_colors.get(zoom_model, None)

    for layer_idx, ax_sub in zip(zoom_layers, ax_bottoms):
        y_val = zoom_values[layer_idx]

        # Highlight point on top plot with a square marker
        ax_top.plot(
            layer_idx, y_val,
            marker='s', markersize=20, 
            markerfacecolor='none', markeredgecolor=model_color, markeredgewidth=4,
            clip_on=False
        )
        
        # Connection lines from point to top-left and top-right of subplot axes
        con1 = ConnectionPatch(
            xyA=(layer_idx, y_val), coordsA=ax_top.transData,
            xyB=(0, 1), coordsB=ax_sub.transAxes,
            linestyle="--", color=model_color, alpha=0.6, linewidth=1.5,
        )
        con2 = ConnectionPatch(
            xyA=(layer_idx, y_val), coordsA=ax_top.transData,
            xyB=(1, 1), coordsB=ax_sub.transAxes,
            linestyle="--", color=model_color, alpha=0.6, linewidth=1.5,
        )
        
        fig.add_artist(con1)
        fig.add_artist(con2)