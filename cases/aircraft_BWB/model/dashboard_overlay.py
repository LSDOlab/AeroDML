import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from dashboard import connect, history_kind, make_history_labels, HERE, PAPER_HISTORIES

OUTPUT_DIR = HERE / "plots"


COLORS = [
    "#5B2A86",
    "#1B7F79",
    "#C44569",
    "#6B8E23",
    "#8C564B",
    "#7A4EAB",
    "#3D5A40",
    "#B56576",
]


def lighten_color(color, amount=0.45):
    """Blend a color toward white. amount=0 returns original, 1 returns white."""
    rgb = np.array(to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = (1 - amount) * rgb + amount * white
    return to_hex(mixed)


def add_run_color_legend(ax, labels, show_validation_legend=False):
    """Show model colors, or separate entries for plotted training/validation curves."""
    if show_validation_legend:
        handles = []
        for i, label in enumerate(labels):
            base_color = COLORS[i % len(COLORS)]
            for split, color, linestyle in (
                ("training", base_color, "-"),
                ("validation", lighten_color(base_color, amount=0.45), "--"),
            ):
                line = next(
                    (line for line in ax.lines
                     if to_hex(line.get_color()) == to_hex(color)
                     and line.get_linestyle() == linestyle),
                    None,
                )
                if line is not None:
                    handles.append(Line2D(
                        [], [], color=color, linestyle=linestyle,
                        linewidth=line.get_linewidth(), alpha=line.get_alpha(),
                        label=f"{label} ({split})",
                    ))
        if handles:
            ax.legend(handles=handles, fontsize="small")
        return

    handles = [
        Patch(
            facecolor=COLORS[i % len(COLORS)],
            edgecolor="none",
            label=label,
        )
        for i, label in enumerate(labels)
    ]
    if handles:
        ax.legend(handles=handles, fontsize="small")


def get_history_cols(df):
    cols = [c for c in df.columns if c.startswith("aux_history_")]
    if not cols:
        raise ValueError("No aux_history_* columns found.")
    return cols


def metric_series(df, col, metric="loss"):
    values = []
    for x in df[col]:
        if isinstance(x, dict) and metric in x:
            values.append(x[metric])
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def average_metric(df, cols, metric="loss"):
    if not cols:
        return None
    stacked = np.vstack([metric_series(df, col, metric=metric) for col in cols])
    return np.nanmean(stacked, axis=0)


def force_abs_error_series(df, col, force):
    """Absolute lift/drag error for one aux_history column at each iteration."""
    pred_key = f"{force}_pred"
    data_key = f"{force}_data"
    values = []

    for x in df[col]:
        if isinstance(x, dict) and pred_key in x and data_key in x:
            values.append(abs(float(x[pred_key]) - float(x[data_key])))
        else:
            values.append(np.nan)

    return np.asarray(values, dtype=float)


def average_force_mae(df, cols, force):
    """Mean absolute error across the selected training/validation histories."""
    if not cols:
        return None

    stacked = np.vstack([force_abs_error_series(df, col, force) for col in cols])
    counts = np.sum(np.isfinite(stacked), axis=0)
    mae = np.full(stacked.shape[1], np.nan, dtype=float)
    np.divide(np.nansum(stacked, axis=0), counts, out=mae, where=counts > 0)

    return mae if np.isfinite(mae).any() else None


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def plot_overlays(
    dfs,
    labels,
    metric="loss",
    average_validation=True,
    average_training=False,
    show_validation=True,
    show_training=True,
    show_validation_legend=False,
):
    if len(dfs) != len(labels):
        raise ValueError("dfs and labels must have the same length")
    if not show_training and not show_validation:
        raise ValueError("At least one of show_training or show_validation must be True")

    fig, ax = plt.subplots(figsize=(8, 6))

    for run_idx, (df, run_label) in enumerate(zip(dfs, labels)):
        base_color = COLORS[run_idx % len(COLORS)]
        val_color = lighten_color(base_color, amount=0.45)

        iterations = df["iter_history"].to_numpy()
        history_cols = get_history_cols(df)
        history_labels = make_history_labels(history_cols)

        train_cols = [c for c in history_cols if history_kind(c) == "training"]
        val_cols = [c for c in history_cols if history_kind(c) == "validation"]

        # Training lines use the base run color.
        if show_training:
            if average_training:
                train_avg = average_metric(df, train_cols, metric=metric)
                if train_avg is not None:
                    ax.semilogy(
                        iterations,
                        train_avg,
                        label="_nolegend_",
                        linewidth=1.8,
                        color=base_color,
                    )
            elif train_cols:
                col = train_cols[0]
                val_train = metric_series(df, col, metric=metric)
                ax.semilogy(
                    iterations,
                    val_train,
                    label="_nolegend_",
                    linewidth=1.8,
                    alpha=0.95,
                    color=base_color,
                )

        # Validation uses the lighter run color; dashed style distinguishes the split.
        if show_validation:
            if average_validation and val_cols:
                val_val = average_metric(df, val_cols, metric=metric)
                ax.semilogy(
                    iterations,
                    val_val,
                    "--",
                    label="_nolegend_",
                    linewidth=1.8,
                    color=val_color,
                )
            else:
                for col in val_cols:
                    ax.semilogy(
                        iterations,
                        metric_series(df, col, metric=metric),
                        "--",
                        label="_nolegend_",
                        linewidth=2.0,
                        alpha=0.95,
                        color=val_color,
                    )

        # print(metric, run_label, len(iterations), len(val_train))
        iter_lim = 100_000
        # iter_lim = 35_000
        # print('validation ', metric, run_label, min(v for i, v in zip(iterations, val_val) if i < iter_lim))
        # print('train      ', metric, run_label, min(v for i, v in zip(iterations, val_train) if i < iter_lim))

    ax.set_title("Loss History" if metric == "loss" else f"{metric} history")
    ax.set_xlabel("Iteration")
    # ax.set_xlim(right=20_000)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    add_run_color_legend(ax, labels, show_validation_legend=show_validation_legend)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"overlay_{metric}.pdf", dpi=200)
    plt.show()


def plot_lift_drag_mae_overlays(
    dfs,
    labels,
    average_validation=True,
    average_training=True,
    show_validation=True,
    show_training=True,
    show_validation_legend=False,
):
    """Plot Lift and Drag MAE histories for multiple runs on one two-panel figure."""
    if len(dfs) != len(labels):
        raise ValueError("dfs and labels must have the same length")
    if not show_training and not show_validation:
        raise ValueError("At least one of show_training or show_validation must be True")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, force, title in zip(axes, ("L", "D"), ("Lift MAE", "Drag MAE")):
        plotted_any = False

        for run_idx, (df, run_label) in enumerate(zip(dfs, labels)):
            base_color = COLORS[run_idx % len(COLORS)]
            val_color = lighten_color(base_color, amount=0.45)
            iterations = df["iter_history"].to_numpy()

            history_cols = get_history_cols(df)
            history_labels = make_history_labels(history_cols)
            train_cols = [c for c in history_cols if history_kind(c) == "training"]
            val_cols = [c for c in history_cols if history_kind(c) == "validation"]

            if show_training:
                if average_training:
                    train_mae = average_force_mae(df, train_cols, force)
                    if train_mae is not None:
                        ax.semilogy(
                            iterations,
                            train_mae,
                            label="_nolegend_",
                            linewidth=1.8,
                            color=base_color,
                        )
                        plotted_any = True
                else:
                    for col in train_cols:
                        train_mae = force_abs_error_series(df, col, force)
                        if np.isfinite(train_mae).any():
                            ax.semilogy(
                                iterations,
                                train_mae,
                                label="_nolegend_",
                                linewidth=1.8,
                                alpha=0.95,
                                color=base_color,
                            )
                            plotted_any = True

            if show_validation:
                if average_validation:
                    val_mae = average_force_mae(df, val_cols, force)
                    if val_mae is not None:
                        ax.semilogy(
                            iterations,
                            val_mae,
                            "--",
                            label="_nolegend_",
                            linewidth=1.8,
                            color=val_color,
                        )
                        plotted_any = True
                else:
                    for col in val_cols:
                        val_mae = force_abs_error_series(df, col, force)
                        if np.isfinite(val_mae).any():
                            ax.semilogy(
                                iterations,
                                val_mae,
                                "--",
                                label="_nolegend_",
                                linewidth=1.8,
                                alpha=0.95,
                                color=val_color,
                            )
                            plotted_any = True

        ax.set_title(f"{title} history")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            add_run_color_legend(ax, labels, show_validation_legend=show_validation_legend)
        else:
            ax.text(
                0.5,
                0.5,
                f"No {force}_pred / {force}_data values found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.suptitle("Lift / Drag MAE history")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "overlay_lift_drag_mae.pdf", dpi=200)
    plt.show()


if __name__ == "__main__":
    final_overlay_labels = [label for label, history in PAPER_HISTORIES.values()]
    final_overlays = [
        connect(history, folder=str(HERE / "training_stats"))
        for label, history in PAPER_HISTORIES.values()
    ]

    # metrics:
    metrics = ["loss", 'res_vx', 'res_vy', 'res_vz', 'res_T', 'res_p', 'res_nuTilda']
    show_training = True
    show_validation_legend = True  # False restores one legend entry per model.
    # metrics = ["loss"]
    for metric in metrics:
        plot_overlays(
                final_overlays,
                final_overlay_labels,
                metric=metric,             # residual / loss key inside each aux_history_* dict
                average_validation=True,   # average validation histories when shown
                average_training=True,     # average training histories when shown
                show_validation=True,      # set False to hide validation curves
                show_validation_legend=show_validation_legend,
                show_training=show_training,        # set False to hide training curves
        )

    plot_lift_drag_mae_overlays(
        final_overlays,
        final_overlay_labels,
        average_validation=True,
        average_training=True,
        show_validation=True,      # set False to hide validation MAE
        show_validation_legend=show_validation_legend,
        show_training=show_training,        # set False to hide training MAE
    )
