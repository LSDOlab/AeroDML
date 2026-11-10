"""Evaluate the three release test cases with a selected saved model."""
import argparse

MODELS = {
    "rans": ("RANS_GNN_trained_paper.pickle", False),
    "panel": ("PG_RANS_GNN_trained_paper.pickle", "panel"),
    "euler": ("EG_RANS_GNN_trained_paper.pickle", "euler"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="rans")
    args = parser.parse_args()
    from run import main as evaluate
    checkpoint, fidelity = MODELS[args.model]
    evaluate(
        load=True, train=False, save=False,
        dataset_dir="sample_dataset", split="sample_split",
        load_model_path=checkpoint, as_multifidelity_model=fidelity,
        custom_vertices=False, model_name=f"sample_{args.model}_gnn",
        exclude_validation_cases=False,
    )


if __name__ == "__main__":
    main()
