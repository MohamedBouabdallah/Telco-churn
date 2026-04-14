import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Styling constants
COLOR_PALETTE = [
    "#0C1618", "#E6D06A", "#004643", "#F6BE9A",
    "#7D9D8B", "#D1AC00", "#FAF4D3", "#D9D9D9"
]
PRIMARY_COLOR = COLOR_PALETTE[0]
LIGHT_GREY = "#D9D9D9"


def set_theme():
    """Apply a consistent global visual style for all plots."""
    sns.set_theme(style="whitegrid")
    sns.set_palette(COLOR_PALETTE)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.family": ["sans-serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.color": LIGHT_GREY,
        "grid.linestyle": (0, (4, 6)),
        "grid.linewidth": 0.8,
        "axes.grid.axis": "y",
    })


def clean_axes(ax):
    """Simplify plot aesthetics by removing unnecessary spines and styling axes."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, labelleft=True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(PRIMARY_COLOR)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save_figure(fig, save_path=None, dpi=150, bbox_inches="tight", close=False):
    """Save a matplotlib figure if a path is provided."""
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to: {save_path}")

    if close:
        plt.close(fig)


def bar_plot(
    data,
    column,
    title="",
    normalize=False,
    order=None,
    save_path=None,
    dpi=150,
    close=False,
):
    """Plot the distribution of a categorical variable as a bar chart."""
    counts = data[column].value_counts(normalize=normalize)

    if order is not None:
        counts = counts.reindex(order)

    counts = counts.reset_index()
    counts.columns = [column, "value"]

    fig, ax = plt.subplots()
    sns.barplot(data=counts, x=column, y="value", ax=ax)
    clean_axes(ax)

    for index, row in counts.iterrows():
        label = f"{row['value']:.1%}" if normalize else f"{int(row['value'])}"
        ax.text(index, row["value"], label, ha="center", va="bottom")

    ax.set(title=title, xlabel="", ylabel="")
    plt.tight_layout()

    save_figure(fig, save_path=save_path, dpi=dpi, close=close)
    return ax


def prepare_stacked_data(data, target, column):
    """Compute normalized proportions of a categorical variable within each target group."""
    return (
        data.groupby(target)[column]
        .value_counts(normalize=True)
        .rename("prop")
        .reset_index()
    )


def stacked_bar(
    data,
    target,
    column,
    title="",
    save_path=None,
    dpi=150,
    close=False,
):
    """Plot a stacked bar chart comparing category distributions across target groups."""
    plot_data = prepare_stacked_data(data, target, column)
    pivot_df = plot_data.pivot(index=target, columns=column, values="prop")

    ax = pivot_df.plot(kind="bar", stacked=True, figsize=(8, 5))
    fig = ax.get_figure()

    clean_axes(ax)
    ax.set(title=title, xlabel="", ylabel="Proportion")
    ax.legend(title=column, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_figure(fig, save_path=save_path, dpi=dpi, close=close)
    return ax


def kde_plot(
    data,
    x,
    hue=None,
    title="",
    palette=None,
    figsize=(8, 5),
    save_path=None,
    dpi=150,
    close=False,
):
    """Plot a smoothed density curve of a numerical variable, optionally split by groups."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(
        data=data,
        x=x,
        hue=hue,
        fill=True,
        palette=palette,
        ax=ax
    )
    ax.set_title(title)
    clean_axes(ax)
    plt.tight_layout()

    save_figure(fig, save_path=save_path, dpi=dpi, close=close)
    return ax


def hist_plot(
    data,
    column,
    bins=30,
    title="",
    xlabel="",
    ylabel="Count",
    vlines=None,
    figsize=(8, 5),
    save_path=None,
    dpi=150,
    close=False,
):
    """Plot a histogram for a numerical variable."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data[column], bins=bins)

    if vlines is not None:
        for line in vlines:
            ax.axvline(
                line["x"],
                linestyle=line.get("linestyle", "--"),
                label=line.get("label", None),
            )

    clean_axes(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else column)
    ax.set_ylabel(ylabel)

    if vlines is not None:
        ax.legend()

    plt.tight_layout()

    save_figure(fig, save_path=save_path, dpi=dpi, close=close)
    return ax