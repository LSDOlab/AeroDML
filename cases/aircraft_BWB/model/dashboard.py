import os
import pandas as pd

def connect(*names, folder="training_stats"):
    dfs = []
    offset = 0

    for i, name in enumerate(names):
        print(f"reading case {name}, \t iter start: {offset}")
        if name is not None:
            filename = os.path.join(folder, f"{name}.jsonl")
        else:
            filename = 'training_aux_info.jsonl'

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

        df = pd.read_json(filename, lines=True).copy()

        # Find all aux_history_* columns automatically
        history_cols = sorted(
            [col for col in df.columns if col.startswith("aux_history_")],
            key=history_sort_key,
        )

        if not history_cols:
            raise ValueError("No aux_history_* columns were found.")

        history_labels = make_history_labels(history_cols)
        pretty_print_recent(df, history_cols, history_labels, n=3)

        # Normalize so each file starts from 0, then shift by offset
        df["iter_history"] = df["iter_history"] - df["iter_history"].iloc[0] + offset

        dfs.append(df)
        offset = int(df["iter_history"].iloc[-1])
    return pd.concat(dfs, ignore_index=True)

def pretty_print_recent(df, history_cols, history_labels, n=3):
    def build_group(metric_kind):
        cols = []
        for col in history_cols:
            sample = df[col].iloc[0]

            if metric_kind == "loss" and "loss" in sample:
                cols.append((f"{history_labels[col]} loss", col, "loss"))
            elif metric_kind == "L_err" and all(k in sample for k in ("L_pred", "L_data")):
                cols.append((f"{history_labels[col]} L err %", col, "L_err"))
            elif metric_kind == "D_err" and all(k in sample for k in ("D_pred", "D_data")):
                cols.append((f"{history_labels[col]} D err %", col, "D_err"))

        return cols

    loss_cols = build_group("loss")
    lift_cols = build_group("L_err")
    drag_cols = build_group("D_err")

    groups = [
        [("iter", None, None)],
        loss_cols,
        lift_cols,
        drag_cols,
    ]
    groups = [g for g in groups if g]

    def value_for(row, kind, col):
        if kind is None:
            return f"{int(row['iter_history'])}"
        if kind == "loss":
            return fmt_value(row[col]["loss"])
        if kind == "L_err":
            return fmt_pct(rel_pct_err(row[col]["L_pred"], row[col]["L_data"]))
        if kind == "D_err":
            return fmt_pct(rel_pct_err(row[col]["D_pred"], row[col]["D_data"]))
        return ""

    def build_row_groups(row, iter_override=None):
        row_groups = []
        for group in groups:
            vals = []
            for _, col, kind in group:
                if kind is None and iter_override is not None:
                    vals.append(iter_override)
                else:
                    vals.append(value_for(row, kind, col))
            row_groups.append(vals)
        return row_groups

    group_headers = [[header for header, _, _ in group] for group in groups]

    group_rows = []


    # Add one extra bottom row for the minimum global save_metric, if present
    best_save_metric = None
    if "save_metric" in df.columns and df["save_metric"].notna().any():
        best_idx = df["save_metric"].idxmin()
        best_row = df.loc[best_idx]
        best_save_metric = df.loc[best_idx, "save_metric"]
        group_rows.append(build_row_groups(best_row, iter_override=f"{int(best_row['iter_history'])}*"))

    for _, row in df.tail(n).iterrows():
        group_rows.append(build_row_groups(row))

    group_widths = []
    for g, headers in enumerate(group_headers):
        widths = []
        for i in range(len(headers)):
            cell_width = max(
                len(headers[i]),
                max(len(row_groups[g][i]) for row_groups in group_rows)
            )
            widths.append(cell_width)
        group_widths.append(widths)

    def format_group(values, widths):
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    print(f"\nLast {n} entries:")

    header_line = "  ||  ".join(
        format_group(group_headers[g], group_widths[g])
        for g in range(len(groups))
    )
    print(header_line)

    divider_line = "==||==".join(
        "-+-".join("-" * w for w in group_widths[g])
        for g in range(len(groups))
    )
    print(divider_line)

    for row_groups in group_rows:
        row_line = "  ||  ".join(
            format_group(row_groups[g], group_widths[g])
            for g in range(len(groups))
        )
        print(row_line)

    # if best_save_metric is not None:
    #     print(f"\n* bottom row = min save_metric ({fmt_value(best_save_metric)})")

    print()

def history_kind(col_name):
    if "_tr" in col_name or "_th" in col_name:
        return "training"
    if "_te" in col_name:
        return "validation"
    return "other"

def history_sort_key(col_name):
    order = {"training": 0, "validation": 1, "other": 2}
    return (order[history_kind(col_name)], col_name)

def make_history_labels(history_cols):
    counts = {"training": 0, "validation": 0, "other": 0}
    labels = {}

    for col in history_cols:
        kind = history_kind(col)
        counts[kind] += 1

        if kind == "training":
            base = "Train"
        elif kind == "validation":
            base = "Val."
        else:
            base = col.replace("aux_history_", "")

        labels[col] = base if counts[kind] == 1 else f"{base} {counts[kind]}"

    return labels

def fmt_value(val):
    return f"{val:.5g}"

def fmt_pct(val):
    return f"{val:.2f}%"

def rel_pct_err(pred, data, eps=1e-12):
    return 100.0 * abs(pred - data) / max(abs(data), eps)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    tsd = 'training_stats'

    # sampling_200
    stats_pg_rgnn = [f'{tsd}/training_aux_info_PG_RANS_GNN']
    stats_eg_rgnn = [f'{tsd}/training_aux_info_EG_RANS_GNN', f'{tsd}/training_aux_info_EG_RANS_GNN_2', f'{tsd}/training_aux_info_EG_RANS_GNN_3']
    stats_rgnn = [f'{tsd}/training_aux_info_RANS_GNN', f'{tsd}/training_aux_info_RANS_GNN_2', f'{tsd}/training_aux_info_RANS_GNN_3']

    # sample_b
    stats_pg_rgnn = [f'{tsd}/training_aux_info_PG_RANS_GNN_sample_b',f'{tsd}/training_aux_info_PG_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_PG_RANS_GNN_tune_sample_b']
    stats_eg_rgnn = [f'{tsd}/training_aux_info_EG_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_EG_RANS_GNN_tune_sample_b']
    stats_rgnn = [f'{tsd}/training_aux_info_RANS_GNN_3_sample_b', f'{tsd}/training_aux_info_RANS_GNN_tune_sample_b']

    connections = [
        *stats_pg_rgnn,
        *stats_eg_rgnn,
        *stats_rgnn,
    ]
    df = connect(*connections, folder="")

    # Find all aux_history_* columns automatically
    history_cols = sorted(
        [col for col in df.columns if col.startswith("aux_history_")],
        key=history_sort_key,
    )

    if not history_cols:
        raise ValueError("No aux_history_* columns were found.")

    history_labels = make_history_labels(history_cols)

    history_palette = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#B279A2",  # purple
        "#72B7B2",  # teal
        "#FF9DA6",  # pink
        "#9D755D",  # brown
    ]
    history_colors = {
        col: history_palette[i % len(history_palette)]
        for i, col in enumerate(history_cols)
    }

    drag_pred_color = "#56B4E9"
    drag_data_color = "#0072B2" 
    lift_pred_color = "#E69F00"
    lift_data_color = "#D55E00"

    iterations = df["iter_history"]

    # Collect metric names from the first history, keeping only keys common to all histories
    metric_names = list(df[history_cols[0]].iloc[0].keys())
    metric_names = [name for name in metric_names if all(name in df[col].iloc[0] for col in history_cols)]
    metric_names = [name for name in metric_names if not ('L_' in name or 'D_' in name)]
    metric_names = [name for name in metric_names if not (
        'pinn_' in name or
        'data_loss' in name or
        'mom_' in name or
        'turb' in name or
        'uity' in name or
        'rgy' in name
    )]

    # Organize metrics for all histories
    all_metrics = {
        col: {name: [] for name in metric_names}
        for col in history_cols
    }

    for col in history_cols:
        for aux_history in df[col]:
            for name in metric_names:
                all_metrics[col][name].append(aux_history[name])

    # Only make lift/drag plots for histories that actually contain those keys
    lift_drag_cols = [
        col for col in history_cols
        if all(k in df[col].iloc[0] for k in ("D_data", "D_pred", "L_data", "L_pred"))
    ]

    # Make subplot grid automatically
    num_metrics = len(metric_names) + len(lift_drag_cols)
    ncols = math.ceil(math.sqrt(num_metrics))
    nrows = math.ceil(num_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    # Metric plots
    for i, name in enumerate(metric_names):
        ax = axes[i]
        for col in history_cols:
            linestyle = "--" if history_kind(col) == "validation" else "-"
            ax.semilogy(
                iterations,
                all_metrics[col][name],
                label=history_labels[col],
                linestyle=linestyle,
                color=history_colors[col],
                linewidth=2.0,
                alpha=0.95,
            )
        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    # Lift / Drag plots: one subplot per history
    for j, col in enumerate(lift_drag_cols):
        aux_list = df[col]

        D_data = [x["D_data"] for x in aux_list]
        D_pred = [x["D_pred"] for x in aux_list]
        L_data = [x["L_data"] for x in aux_list]
        L_pred = [x["L_pred"] for x in aux_list]

        ax = axes[len(metric_names) + j]
        ax.plot(iterations, L_pred, label="L_pred", color=lift_pred_color, linewidth=2.0)
        ax.plot(iterations, D_pred, label="D_pred", color=drag_pred_color, linewidth=2.0)
        ax.plot(iterations, D_data, "--", label="D_data", color=drag_data_color, linewidth=2.0)
        ax.plot(iterations, L_data, "--", label="L_data", color=lift_data_color, linewidth=2.0)

        # y-limits based only on DATA lift/drag, with 20% padding
        data_min = min(min(D_data), min(L_data))
        data_max = max(max(D_data), max(L_data))
        pad = 0.2 * (data_max - data_min if data_max > data_min else 1.0)
        # ax.set_ylim(data_min - pad, data_max + pad)

        ax.set_title(f"{history_labels[col]}: Lift / Drag")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # save as image or show gui
    if 0:
        plt.show()
    else:
        plt.savefig("plots/training_metrics.png")