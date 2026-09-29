import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

from dashboard import connect, history_kind, make_history_labels


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


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def plot_overlays(dfs, labels, metric="loss", average_validation=True, average_training=False):
    if len(dfs) != len(labels):
        raise ValueError("dfs and labels must have the same length")

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
        if average_training:
            train_avg = average_metric(df, train_cols, metric=metric)
            if train_avg is not None:
                ax.semilogy(
                    iterations,
                    train_avg,
                    label=f"{run_label} train avg",
                    linewidth=2.4,
                    color=base_color,
                )
        else:
            col = train_cols[0]
            val_train = metric_series(df, col, metric=metric)
            line_label = f"{run_label} {history_labels.get(col, col)}"
            ax.semilogy(
                iterations,
                val_train,
                label=line_label,
                linewidth=1.8,
                alpha=0.95,
                color=base_color,
            )

        # Validation uses a lighter version of the same run color.
        if average_validation and val_cols:
            val_val = average_metric(df, val_cols, metric=metric)
            ax.semilogy(
                iterations,
                val_val,
                "--",
                label=f"{run_label} Validation",
                linewidth=2.8,
                color=val_color,
            )
        else:
            val_val = average_metric(df, val_cols, metric=metric)
            for col in val_cols:
                line_label = f"{run_label} {history_labels.get(col, col)}"
                ax.semilogy(
                    iterations,
                    metric_series(df, col, metric=metric),
                    "--",
                    label=line_label,
                    linewidth=2.0,
                    alpha=0.95,
                    color=val_color,
                )

        # print(metric, run_label, len(iterations), len(val_train))
        iter_lim = 100_000
        # iter_lim = 35_000
        # print('validation ', metric, run_label, min(v for i, v in zip(iterations, val_val) if i < iter_lim))
        # print('train      ', metric, run_label, min(v for i, v in zip(iterations, val_train) if i < iter_lim))

    # just concatenate the models so the models are in the title
    model_string = " ".join(labels)
    ax.set_title(f"{model_string} convergence comparison")
    # ax.set_title(f"{metric} convergence")
    ax.set_xlabel("Iteration")
    # ax.set_xlim(right=20_000)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(f"plots/overlay_{metric}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    tsd = "training_stats"

    # sampling_200 connections
    pg_rgnn_data = connect(*[f'{tsd}/training_aux_info_PG_RANS_GNN'], folder="")
    eg_rgnn_data = connect(*[f'{tsd}/training_aux_info_EG_RANS_GNN', f'{tsd}/training_aux_info_EG_RANS_GNN_2', f'{tsd}/training_aux_info_EG_RANS_GNN_3'], folder="")
    rgnn_data = connect(*[f'{tsd}/training_aux_info_RANS_GNN', f'{tsd}/training_aux_info_RANS_GNN_2', f'{tsd}/training_aux_info_RANS_GNN_3'], folder="")

    # sample_b
    pg_rgnn_data = connect(*[f'{tsd}/training_aux_info_PG_RANS_GNN_sample_b', f'{tsd}/training_aux_info_PG_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_PG_RANS_GNN_tune_sample_b'], folder="")
    eg_rgnn_data = connect(*[f'{tsd}/training_aux_info_EG_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_EG_RANS_GNN_tune_sample_b'], folder="")
    rgnn_data = connect(*[f'{tsd}/training_aux_info_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_RANS_GNN_tune_sample_b'], folder="")

    # overlays
    final_overlays = [eg_rgnn_data, rgnn_data, pg_rgnn_data] #+ overlays
    final_overlay_labels = ["EG-RANS-GNN", "RANS-GNN", "PG-RANS-GNN"] #+ overlay_labels

    # metrics:
    metrics = ["loss", 'res_vx', 'res_vy', 'res_vz', 'res_T', 'res_p', 'res_nuTilda']
    # metrics = ["loss"]
    for metric in metrics:
        plot_overlays(
                final_overlays,
                final_overlay_labels,
                metric=metric,            # residual / loss key inside each aux_history_* dict
                average_validation=True,   # one dashed validation line per run
                average_training=False,    # keep individual training curves
        )
