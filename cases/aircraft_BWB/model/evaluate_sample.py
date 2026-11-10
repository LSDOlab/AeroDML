"""Compare saved pressure plots for a selected release test case."""

CASE_ID = 38  # Choose 38, 132, or 155.

MODELS = {
    "rans": ("RANS_GNN_trained_paper.pickle", False),
    "panel": ("PG_RANS_GNN_trained_paper.pickle", "panel"),
    "euler": ("EG_RANS_GNN_trained_paper.pickle", "euler"),
}


def main():
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Uncomment to regenerate model outputs before assembling the comparison.
    from run import main as evaluate
    for name, (checkpoint, fidelity) in MODELS.items():
        print(f"Evaluating {name} model...")
        evaluate(
            load=True, train=False, save=False,
            dataset_dir="sample_dataset", split="sample_split",
            load_model_path=checkpoint, as_multifidelity_model=fidelity,
            custom_vertices=False, model_name=f"sample_{name}_gnn",
            exclude_validation_cases=False,
        )

    # ==================== Create comparison plot similar to the one in the readme ====================
    root = Path(__file__).resolve().parent / "post_processing_plots"
    if not all((root / f"sample_{name}_gnn").is_dir() for name in MODELS):
        print("Generate the RANS, panel, and Euler output directories first.")
        return
    columns = [("rans", "CFD", "RANS", "black")]
    labels = [("rans", "RANS-GNN", "#0072BD"), ("panel", "PG-RANS-GNN", "#C99700"), ("euler", "EG-RANS-GNN", "#46642D")]
    columns += [(name, kind, label, color) for kind in ("NN", "error") for name, label, color in labels]
    views = [("surface_{kind}_p_xc_windows", (0.10, 0.85, 0.12, 0.80)), ("xc_{kind}_y16_p", (0.12, 0.86, 0.02, 0.98))]
    fig, axes = plt.subplots(2, 7, figsize=(24, 8), gridspec_kw={"height_ratios": [1, 0.65]})
    try:
        for col, (name, kind, label, color) in enumerate(columns):
            folder = root / f"sample_{name}_gnn/field_plots/case_{CASE_ID}" / kind
            for row, (pattern, crop) in enumerate(views):
                prefix = pattern.format(kind=kind)
                image = plt.imread(folder / f"{prefix}_sample_{name}_gnn_case_{CASE_ID}.png")
                height, width = image.shape[:2]
                top, bottom, left, right = crop
                axes[row, col].imshow(image[int(top * height):int(bottom * height), int(left * width):int(right * width)])
                axes[row, col].set_axis_off()
            axes[0, col].set_title(label, color=color, fontfamily="serif", fontweight="bold", fontsize=23)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.21, top=0.82, wspace=0.02, hspace=0.02)
        for start, end, title in ((0, 0, "Reference"), (1, 3, "Predicted Pressure Fields"), (4, 6, "Pressure Error Fields")):
            left, right = axes[0, start].get_position().x0, axes[0, end].get_position().x1
            fig.text((left + right) / 2, 0.96, title, ha="center", va="top", fontfamily="serif", fontsize=31)
            fig.add_artist(plt.Line2D([left, right], [0.88, 0.88], transform=fig.transFigure, color="#18243B"))
        # HARDCODED: Match plot_field's saved jet/coolwarm colorbars, using this case's freestream.
        cases_path = root.parent.parent.parent / "datasets/BWB/data/sample_dataset/cases.json"
        cases = json.loads(cases_path.read_text())["cases"]
        flow = next(case["flow_conditions"] for case in cases if str(case["key"]) == str(CASE_ID))
        p_inf, temperature, velocity = (flow[key]["value"] for key in ("p_inf", "T_inf", "v_inf"))
        q_inf = 0.5 * p_inf / (287.05 * temperature) * np.dot(velocity, velocity)
        bars = [(0.10, "jet", (np.array([1e4, 4e4]) - p_inf) / q_inf, r"$C_p$"),
                (0.61, "coolwarm", np.array([-3e3, 3e3]) / q_inf, r"$C_p^{\mathrm{pred}} - C_p^{\mathrm{RANS}}$")]
        for left, cmap, limits, label in bars:
            cax = fig.add_axes([left, 0.075, 0.34, 0.028])
            scalar = plt.cm.ScalarMappable(norm=plt.Normalize(*limits), cmap=cmap)
            bar = fig.colorbar(scalar, cax=cax, orientation="horizontal", ticks=np.linspace(*limits, 5), format="%.2f")
            bar.ax.set_title(label, fontsize=28, pad=14, math_fontfamily="cm")
            bar.ax.tick_params(labelsize=22)
            plt.setp(bar.ax.get_xticklabels(), fontfamily="serif")
        output = root / f"pressure_comparison_case_{CASE_ID}.png"
        fig.savefig(output, dpi=200, facecolor="white")
        print(f"Saved {output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
