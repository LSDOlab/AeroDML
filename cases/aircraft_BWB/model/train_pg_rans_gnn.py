from run import main

if __name__ == "__main__":
    # ============================ Important Script Flags ============================:

    load, train, save = False, True, True         # Train from scratch
    load_model_path = None
    save_training_path = 'PG_RANS_GNN'

    # Low-fidelity model input:
    as_multifidelity_model = 'panel' # ALWAYS TRUE FOR THIS SCRIPT

    # DATASET
    # dataset_dir, split = 'sampling_200', 'split_0'
    dataset_dir, split = 'sample_case', 'split_0'

    save_training_path = save_training_path + '_' + dataset_dir
    print("***RUNNING TRAIN_MF_GNN.PY***")
    main(
        load=load, 
        train=train, 
        save=save,
        dataset_dir = dataset_dir,
        split = split,
        load_model_path=load_model_path,
        save_training_path=save_training_path,
        as_multifidelity_model=as_multifidelity_model,
        with_pinns=False,
    )