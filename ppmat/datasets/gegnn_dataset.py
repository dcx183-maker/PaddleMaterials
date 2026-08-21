# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import pickle
from typing import Optional

import numpy as np
import paddle
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger

__all__ = ["BinaryActivityDataset"]

_ALLOWABLE_ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]
_ALLOWABLE_DEGREES = list(range(11))
_ALLOWABLE_VALENCES = list(range(7))
_ALLOWABLE_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_ALLOWABLE_NUM_H = [0, 1, 2, 3, 4]


def _one_hot_encoding(x, allowable_set):
    return [int(x == s) for s in allowable_set]


def _canonical_atom_feats(atom):
    """Match the upstream DGL-LifeSci ``CanonicalAtomFeaturizer`` (74 dims)."""
    feats = (
        _one_hot_encoding(atom.GetSymbol(), _ALLOWABLE_ATOM_TYPES)
        + _one_hot_encoding(atom.GetDegree(), _ALLOWABLE_DEGREES)
        + _one_hot_encoding(atom.GetImplicitValence(), _ALLOWABLE_VALENCES)
        + [atom.GetFormalCharge()]
        + [atom.GetNumRadicalElectrons()]
        + _one_hot_encoding(atom.GetHybridization(), _ALLOWABLE_HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + _one_hot_encoding(atom.GetTotalNumHs(), _ALLOWABLE_NUM_H)
    )
    return np.array(feats, dtype=np.float32)


_DATA_FILE = "output_binary_with_inf_all.csv"
_SOLVENT_FILE = "solvent_list.csv"
_DATA_URL = (
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
    "thermodynamic_data_of_binary_mixtures/"
)


_MOLECULAR_GRAPH_VOCAB = {
    "atom": {
        "token_to_id": {
            symbol: index for index, symbol in enumerate(_ALLOWABLE_ATOM_TYPES)
        },
        "num_embeddings": len(_ALLOWABLE_ATOM_TYPES),
    },
    "bond": {
        "token_to_id": {
            "NO_BOND": 0,
            "SINGLE": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
            "AROMATIC": 4,
        },
        "num_embeddings": 5,
    },
}

_MOLECULAR_GRAPH_CFG = {
    "__class_name__": "MolecularGraphConverter",
    "__init_params__": {
        "vocab": _MOLECULAR_GRAPH_VOCAB,
        "remove_h": False,
        "add_self_loops": True,
        "edge_mode": "bidirectional",
    },
}


def build_molecular_graph(mol, converter):
    graph = converter(mol)
    if graph is None:
        return None
    graph.node_feat["h"] = np.stack(
        [_canonical_atom_feats(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())],
        axis=0,
    )
    return graph


class BinaryActivityDataset(Dataset):
    """Binary-mixture activity-coefficient dataset for GE-GNN.

    The dataset reads the upstream ``binaryGamma`` CSV and solvent metadata,
    constructs one molecular graph per component, and returns the two logarithmic
    activity coefficients as training labels. ``comp_inter`` applies the upstream
    TPSA-stratified cross-validation protocol; ``system_extra`` keeps each ordered
    solvent pair in a single fold.

    Dataset format:
        ```text
        binary_activity/
        ├── output_binary_with_inf_all.csv
        └── solvent_list.csv
        ```
        The binary-mixture CSV must contain `solv1`, `solv2`, `solv1_x`,
        `solv1_gamma`, and `solv2_gamma`. The solvent CSV is indexed by
        `solvent_id` and must contain `smiles_can`.

    Args:
        path: Path to the binary-mixture CSV. If it does not exist, the CSV is
            downloaded to the shared PaddleMaterials dataset cache and sibling solvent
            metadata is read from the same directory. Defaults to
            `./data/binary_activity/output_binary_with_inf_all.csv`.
        solvent_list_path: Optional path to the solvent metadata CSV. When omitted,
            `solvent_list.csv` is read from the same directory as `path`.
        split_mode: ``all``, ``comp_inter``, or ``system_extra``.
        split_part: ``all``, ``train``, or ``val``.
        fold: Validation fold index when a split mode is enabled.
        num_folds: Number of cross-validation folds.
        seed: Random seed for deterministic fold assignment.
    """

    name = "binary_activity"
    url = _DATA_URL + _DATA_FILE
    solvent_url = _DATA_URL + _SOLVENT_FILE

    _REQUIRED_COLUMNS = {
        "solv1",
        "solv2",
        "solv1_x",
        "solv1_gamma",
        "solv2_gamma",
    }
    _SPLIT_MODES = {"all", "comp_inter", "system_extra"}
    _SPLIT_PARTS = {"all", "train", "val"}

    def __init__(
        self,
        path: str = "./data/binary_activity/output_binary_with_inf_all.csv",
        solvent_list_path: Optional[str] = None,
        split_mode: str = "all",
        split_part: str = "all",
        fold: int = 0,
        num_folds: int = 5,
        seed: int = 2021,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.build_molecule = BuildMolecule(format="smiles")
        self.graph_converter = build_graph_converter(_MOLECULAR_GRAPH_CFG)

        path, solvent_list_path = self._resolve_data_paths(path, solvent_list_path)

        self.path = path
        self.solvent_list_path = solvent_list_path
        if not osp.exists(path):
            raise FileNotFoundError(f"Binary-mixture CSV is unavailable: {path}")
        solvent_list = pd.read_csv(solvent_list_path, index_col="solvent_id")
        self.dataset = pd.read_csv(path, low_memory=False)
        self._validate_dataset_columns()
        self.dataset = self._apply_split(
            split_mode=split_mode,
            split_part=split_part,
            fold=fold,
            num_folds=num_folds,
            seed=seed,
        )
        self.solvent_smiles = solvent_list["smiles_can"].to_dict()

        self.split_mode = split_mode
        self.split_part = split_part
        self.fold = fold
        self.num_folds = num_folds
        self.seed = seed
        self.overwrite = overwrite
        self.cache_path = cache_path or osp.join(
            osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
        )
        self.solvent_data = self._load_or_build_cache()
        logger.info(
            f"Load {len(self.dataset)} binary-mixture samples from {path}"
        )

    def _resolve_data_paths(self, path, solvent_list_path):
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            path = download.get_datasets_path_from_url(self.url)
        if solvent_list_path is None:
            solvent_list_path = osp.join(osp.dirname(path), _SOLVENT_FILE)
        if not osp.exists(solvent_list_path):
            download.get_path_from_url(
                self.solvent_url, osp.dirname(path), decompress=False
            )
        return path, solvent_list_path

    def _validate_dataset_columns(self):
        missing = self._REQUIRED_COLUMNS.difference(self.dataset.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(
                "Binary activity CSV is missing required columns: " f"{missing_columns}"
            )

    @staticmethod
    def _fold_partitions(fold_ids, split_part, fold):
        if split_part == "train":
            return fold_ids != fold
        if split_part == "val":
            return fold_ids == fold
        return np.ones(len(fold_ids), dtype=bool)

    def _apply_split(self, split_mode, split_part, fold, num_folds, seed):
        if split_mode not in self._SPLIT_MODES:
            allowed = ", ".join(sorted(self._SPLIT_MODES))
            raise ValueError(
                f"Unsupported split_mode {split_mode!r}; choose one of {allowed}."
            )
        if split_part not in self._SPLIT_PARTS:
            allowed = ", ".join(sorted(self._SPLIT_PARTS))
            raise ValueError(
                f"Unsupported split_part {split_part!r}; choose one of {allowed}."
            )
        if split_mode == "all":
            if split_part != "all":
                raise ValueError("split_part must be 'all' when split_mode is 'all'.")
            return self.dataset.reset_index(drop=True)
        if num_folds < 2:
            raise ValueError("num_folds must be at least 2 for cross-validation.")
        if not 0 <= fold < num_folds:
            raise ValueError("fold must be in [0, num_folds).")

        if "tpsa_binary_avg" not in self.dataset.columns:
            raise ValueError(
                f"split_mode={split_mode!r} requires the upstream "
                "'tpsa_binary_avg' stratification column."
            )
        if split_mode == "comp_inter":
            fold_ids = self._composition_fold_ids(num_folds, seed)
        else:
            fold_ids = self._system_fold_ids(num_folds, seed)
        partition_mask = self._fold_partitions(fold_ids, split_part, fold)
        return self.dataset.loc[partition_mask].reset_index(drop=True)

    @staticmethod
    def _stratified_fold_ids(labels, num_folds, seed):
        from sklearn.model_selection import StratifiedKFold

        labels = np.asarray(labels)
        fold_ids = np.empty(len(labels), dtype=np.int64)
        splitter = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        for fold_id, (_, val_indices) in enumerate(
            splitter.split(np.arange(len(labels)), labels)
        ):
            fold_ids[val_indices] = fold_id
        return fold_ids

    def _composition_fold_ids(self, num_folds, seed):
        """Reproduce upstream row-level TPSA-stratified cross-validation."""
        return self._stratified_fold_ids(
            self.dataset["tpsa_binary_avg"].to_numpy(), num_folds, seed
        )

    def _system_fold_ids(self, num_folds, seed):
        """Reproduce upstream ordered-system TPSA-stratified cross-validation."""
        systems = self.dataset.groupby(["solv1", "solv2"], sort=True, as_index=False)[
            "tpsa_binary_avg"
        ].mean()
        system_folds = self._stratified_fold_ids(
            systems["tpsa_binary_avg"].to_numpy(), num_folds, seed
        )
        fold_by_system = {
            (row.solv1, row.solv2): int(system_folds[index])
            for index, row in systems.iterrows()
        }
        return np.asarray(
            [
                fold_by_system[(row.solv1, row.solv2)]
                for row in self.dataset[["solv1", "solv2"]].itertuples(index=False)
            ],
            dtype=np.int64,
        )

    def _load_or_build_cache(self):
        config_path = osp.join(self.cache_path, "build_graph_cfg.pkl")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        config = {
            "solvent_list_path": osp.abspath(self.solvent_list_path),
            "feature_dim": 74,
            "add_self_loops": True,
            "remove_h": False,
        }
        cache_exists = osp.exists(config_path) and osp.isdir(graph_cache_path)
        if cache_exists and not self.overwrite:
            cached_config = self._load_from_cache(config_path)
            if cached_config != config:
                raise ValueError(
                    "Cached molecular graph configuration does not match the current "
                    "dataset. Set overwrite=True or use a different cache_path."
                )
            return {
                solvent_id: osp.join(graph_cache_path, f"{solvent_id}.pkl")
                for solvent_id in self.solvent_smiles
            }

        os.makedirs(graph_cache_path, exist_ok=True)
        self._save_to_cache(config_path, config)
        solvent_data = {}
        for solvent_id, smiles in self.solvent_smiles.items():
            cache_file = osp.join(graph_cache_path, f"{solvent_id}.pkl")
            self._save_to_cache(cache_file, self._build_solvent(solvent_id, smiles))
            solvent_data[solvent_id] = cache_file
        return solvent_data

    @staticmethod
    def _save_to_cache(path, data):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    @staticmethod
    def _load_from_cache(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def _build_solvent(self, solvent_id, smiles):
        mol = self.build_molecule(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for solvent {solvent_id}: {smiles}")
        graph = build_molecular_graph(mol, self.graph_converter)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        return {
            "graph": graph,
            "hba": hba,
            "hbd": hbd,
            "intra_hb": min(hba, hbd),
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, paddle.Tensor):
            idx = idx.item()

        data = self.dataset.iloc[idx]
        ids = [data["solv1"], data["solv2"]]
        solv1 = self._load_from_cache(self.solvent_data[ids[0]])
        solv2 = self._load_from_cache(self.solvent_data[ids[1]])

        sample = {
            "g1": solv1["graph"],
            "g2": solv2["graph"],
            "x1": float(data["solv1_x"]),
            "x2": 1.0 - float(data["solv1_x"]),
            "gamma1": float(data["solv1_gamma"]),
            "gamma2": float(data["solv2_gamma"]),
            "intra_hb1": solv1["intra_hb"],
            "intra_hb2": solv2["intra_hb"],
            "inter_hb": min(solv1["hba"], solv2["hbd"])
            + min(solv1["hbd"], solv2["hba"]),
            "empty_solvsys": self.generate_solvsys(1),
        }
        return sample

    @staticmethod
    def generate_solvsys(batch_size=5):
        n_solv = 2
        num_nodes = n_solv * batch_size

        src_list = []
        dst_list = []

        src = list(range(batch_size))
        dst = list(range(batch_size, n_solv * batch_size))
        src_list.extend(src)
        dst_list.extend(dst)
        src_list.extend(dst)
        dst_list.extend(src)

        for i in range(num_nodes):
            src_list.append(i)
            dst_list.append(i)

        edges = np.array(list(zip(src_list, dst_list)), dtype=np.int64)
        graph = pgl.Graph(
            num_nodes=num_nodes,
            edges=edges,
        )
        return graph
