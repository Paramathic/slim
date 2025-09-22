import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def inch_to_pts(inch):
    return inch * 72.27


def set_size(width, fraction=1, y_scale=1.0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * y_scale)

    return fig_dim


def prepare_figure(size_fraction=1.0):
    if gpu_type == "a100":
        yscale = 1.6
        size = 6.75
    else:
        yscale = 2.0
        size = 3.25
    fig_dim = set_size(inch_to_pts(size), fraction=size_fraction, y_scale=yscale)

    plt.style.use("seaborn-v0_8")
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 6,
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 6,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 1,
        "ytick.labelsize": 4,
        "figure.figsize": fig_dim,
        "lines.linewidth": 0.4,
        "figure.facecolor": "white",
    }

    plt.rcParams.update(tex_fonts)


def post_process_figure(ax):
    # Set default plt figure to ax
    plt.sca(ax)

    plt.grid(True, which="major", color="grey", alpha=0.2, linewidth=0.3)
    plt.grid(
        True, which="minor", color="grey", linestyle="--", alpha=0.1, linewidth=0.1
    )
    plt.minorticks_on()
    plt.tick_params(which="major", axis="y", direction="in", width=0.5, color="grey")
    plt.tick_params(which="minor", axis="y", direction="in", width=0.3, color="grey")
    plt.tick_params(which="major", axis="x", direction="in", width=0.5, color="grey")
    plt.tick_params(which="minor", axis="x", direction="in", width=0.3, color="grey")

    # Remove the upper and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("grey")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("grey")
    ax.spines["left"].set_linewidth(0.5)


# Define the types of layers
def determine_layer_type(row):
    config = row["Configuration"].split(",")
    d_in, d_out = config[2].split("=")[1], config[3].split("=")[1]
    if d_out == d_in:
        return "Self-Attention"
    elif d_out > d_in:
        return "Up-Projection"
    else:
        return "Down-Projection"


def determine_model(row):
    return row["Configuration"].split(" ")[0]


def plot_speedup(data, fig=None, axes=None):
    if gpu_type == "a100":
        angle = 0
        xlabel_fontsize = 10
        title_fontsize = 12
        legend_fontsize = 8
        text_fontsize = 8
        main_title_fontsize = 14
        ylabel_fontsize = 12
    else:
        angle = 40
        xlabel_fontsize = 6
        title_fontsize = 8
        legend_fontsize = 6
        text_fontsize = 6
        main_title_fontsize = 12
        ylabel_fontsize = 8

    # Add a 'Layer Type' column
    data["Layer Type"] = data.apply(determine_layer_type, axis=1)

    # Add a 'Model' column
    data["Model"] = data.apply(determine_model, axis=1)

    # Extract batch sizes
    data["Batch Size"] = data["Configuration"].str.extract(r"bs=(\d+)").astype(int)

    # Get unique models and batch sizes
    models = data["Model"].unique()
    batch_sizes = sorted(data["Batch Size"].unique())
    layer_types = data["Layer Type"].unique()

    # Prepare subplots
    if fig is None or axes is None:
        prepare_figure(size_fraction=1.0)
        fig, axes = plt.subplots(len(models), len(batch_sizes), sharey="row")
        # Colors for the bars - Dark Blue for FP16 LoRA, Red for INT4 LoRA
        colors = ["#2fa7c4", "#d84748"]
        second_plot = False
    else:
        # Choose similar but brighter colors for the second plot
        colors = ["#87d3e0", "#f08a8b"]
        second_plot = True
    if gpu_type == "a100":
        title = "SLiM Speedup on A100-40GB"
    elif gpu_type == "rtx3090":
        title = "SLiM Speedup on RTX 3090"
    elif gpu_type == "rtx3060":
        title = "SLiM Speedup on RTX 3060"
    else:
        raise ValueError(f"Unknown GPU type: {gpu_type}")
    fig.suptitle(title, fontsize=main_title_fontsize)

    # Plot the data
    for i, model in enumerate(models):
        for j, batch_size in enumerate(batch_sizes):
            ax = axes[i, j]
            # Filter data for the specific model and batch size
            subset = data[(data["Model"] == model) & (data["Batch Size"] == batch_size)]
            if not subset.empty:
                fpt16_speedups = []
                int4_speedups = []
                for layer_type in layer_types:
                    layer_subset = subset[subset["Layer Type"] == layer_type]
                    if not layer_subset.empty:
                        fpt16_speedups.append(
                            layer_subset["lora_linear_fp16_speedup"].values[0]
                        )
                        int4_speedups.append(
                            layer_subset["lora_linear_marlin_int4_speedup"].values[0]
                        )
                    else:
                        fpt16_speedups.append(0)
                        int4_speedups.append(0)

                # Plot the bars
                bar_width = 0.45  # Width of the bars

                bars_fp16 = ax.bar(
                    np.arange(len(layer_types)) - bar_width / 2,
                    fpt16_speedups,
                    color=colors[0],
                    width=bar_width,
                    label="FP16 LoRA",
                )
                bars_int4 = ax.bar(
                    np.arange(len(layer_types)) + bar_width / 2,
                    int4_speedups,
                    color=colors[1],
                    width=bar_width,
                    label="INT4 LoRA",
                )

                if not second_plot:
                    # Add value labels on top of the bars
                    for bar in bars_fp16:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"{height:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=text_fontsize,
                            rotation=angle,
                        )

                    for bar in bars_int4:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"{height:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=text_fontsize,
                            rotation=angle,
                        )

            # Set subplot title
            if i == 0:
                ax.set_title(f"Batch Size {batch_size}", fontsize=title_fontsize)

            # Set y-axis label only for the first column
            if j == 0:
                ax.set_ylabel(model, fontsize=ylabel_fontsize)
                if gpu_type != "a100":
                    ax.set_ylim(0, 4.7)

            # Set x-axis labels
            ax.set_xticks([0, 1, 2])
            if i == len(models) - 1:
                ax.set_xticklabels(layer_types, rotation=20, fontsize=xlabel_fontsize)
            else:
                ax.set_xticklabels([])

            post_process_figure(ax)

            ax.set_facecolor("white")

    # Add a legend
    handles = [
        plt.Line2D([0], [0], color=colors[0], label="FP16 LoRA"),
        plt.Line2D([0], [0], color=colors[1], label="INT4 LoRA"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.94),
        ncols=2,
        fontsize=legend_fontsize,
    )

    plt.subplots_adjust(top=0.84)

    # Adjust layout
    # plt.show()
    plt.savefig(f"assets/{gpu_type}_speedup.pdf", bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    # Load the CSV data
    gpu_type = "rtx3060"  # "a100"
    file_path = f"results/{gpu_type}_speedup_results.csv"
    quantization_file_path = f"results/{gpu_type}_speedup_results_quantize_only.csv"
    data = pd.read_csv(file_path)
    plot_breakdown = True
    fig, axes = plot_speedup(data)
    if plot_breakdown:
        quantization_data = pd.read_csv(quantization_file_path)
        plot_speedup(quantization_data, fig, axes)
