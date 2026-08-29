import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_binary_activity_data(
    dataset, split_mode, split_part, fold=0, num_folds=5, seed=2021
):
    if split_mode == "all":
        if split_part != "all":
            raise ValueError("split_part must be 'all' when split_mode is 'all'.")
        return dataset.reset_index(drop=True)

    if split_mode not in {"comp_inter", "system_extra"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    if split_part not in {"train", "val"}:
        raise ValueError(f"Unsupported split_part: {split_part}")
    if "tpsa_binary_avg" not in dataset:
        raise ValueError("Split requires the 'tpsa_binary_avg' column.")
    if not 0 <= fold < num_folds:
        raise ValueError("fold must be in [0, num_folds).")

    splitter = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )
    if split_mode == "comp_inter":
        indices = np.arange(len(dataset))
        fold_ids = np.empty(len(dataset), dtype=np.int64)
        for fold_index, (_, valid_indices) in enumerate(
            splitter.split(indices, dataset["tpsa_binary_avg"])
        ):
            fold_ids[valid_indices] = fold_index
    else:
        systems = dataset.groupby(
            ["solv1", "solv2"], as_index=False
        )["tpsa_binary_avg"].mean()
        system_fold_ids = np.empty(len(systems), dtype=np.int64)
        for fold_index, (_, valid_indices) in enumerate(
            splitter.split(systems, systems["tpsa_binary_avg"])
        ):
            system_fold_ids[valid_indices] = fold_index
        fold_map = {
            (system.solv1, system.solv2): system_fold_ids[index]
            for index, system in systems.iterrows()
        }
        fold_ids = np.asarray(
            [fold_map[(record.solv1, record.solv2)] for record in dataset.itertuples()]
        )

    selected = fold_ids != fold if split_part == "train" else fold_ids == fold
    return dataset.loc[selected].reset_index(drop=True)
