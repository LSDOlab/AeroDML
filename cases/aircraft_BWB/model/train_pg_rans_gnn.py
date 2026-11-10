import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PG_RANS_GNN on your dataset.")
    parser.add_argument("--dataset", required=True, help="Directory under cases/datasets/BWB/data")
    parser.add_argument("--split", required=True, help="Split name, with or without .json")
    args = parser.parse_args()

    # ============================ Important Script Flags ============================:

    load, train, save = False, True, True         # Train from scratch
    load_model_path = None
    save_training_path = 'PG_RANS_GNN'
    custom_vertices = False


    # Low-fidelity model input:
    as_multifidelity_model = 'panel' # Low-fidelity input for this model

    # DATASET
    dataset_dir = args.dataset
    split = args.split.removesuffix(".json")

    # The bundled sample is evaluation-only; training needs a training split.
    data_root = Path(__file__).resolve().parents[2] / "datasets" / "BWB" / "data"
    split_data = json.loads((data_root / dataset_dir / f"{split}.json").read_text())
    if not any(case.get("valid") and case.get("split") == "train" for case in split_data["cases"]):
        parser.error("The selected split has no training cases; the release sample is evaluation-only.")

    from run import main

    save_training_path = save_training_path + '_' + dataset_dir + '_' + split
    print("***RUNNING TRAIN_PG_RANS_GNN.PY***")
    main(
        load=load,
        train=train,
        save=save,
        dataset_dir = dataset_dir,
        split = split,
        load_model_path=load_model_path,
        save_training_path=save_training_path,
        as_multifidelity_model=as_multifidelity_model,
        custom_vertices = custom_vertices,
        with_pinns=False,
    )
