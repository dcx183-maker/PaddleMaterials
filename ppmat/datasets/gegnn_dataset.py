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
from typing import Optional

import numpy as np
import paddle
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.utils import logger
from ppmat.utils.download import get_path_from_url

__all__ = ["BinaryActivityDataset", "BinaryActivityCollator"]

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


_DATA_URL = (
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
    "thermodynamic_data_of_binary_mixtures/"
)


def smiles_to_pgl_graph(smiles, add_self_loop=True):
    """Build a molecular graph with the shared ``BuildMolecule`` factory."""
    mol = BuildMolecule(format="smiles")(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return None

    num_atoms = mol.GetNumAtoms()

    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        for i in range(num_atoms):
            src_list.append(i)
            dst_list.append(i)

    node_feat = np.stack(
        [_canonical_atom_feats(mol.GetAtomWithIdx(i)) for i in range(num_atoms)],
        axis=0,
    )

    edges = np.array(list(zip(src_list, dst_list)), dtype=np.int64)
    graph = pgl.Graph(
        num_nodes=num_atoms,
        edges=edges,
        node_feat={"h": node_feat},
    )
    return graph


class BinaryActivityDataset(Dataset):
    """Binary-mixture activity-coefficient dataset for GE-GNN.

    The dataset reads the upstream ``binaryGamma`` CSV and solvent metadata,
    constructs one molecular graph per component, and returns the two logarithmic
    activity coefficients as training labels. ``comp_inter`` applies the upstream
    TPSA-stratified cross-validation protocol; ``system_extra`` keeps each ordered
    solvent pair in a single fold.

    Args:
        input_file_path: Name or path of the binary-mixture CSV.
        solvent_list_path: Name or path of the solvent metadata CSV.
        data_dir: Directory containing the CSV files. Missing files are downloaded
            from the public PaddleMaterials dataset location.
        split_mode: ``all``, ``comp_inter``, or ``system_extra``.
        split_part: ``all``, ``train``, or ``val``.
        fold: Validation fold index when a split mode is enabled.
        num_folds: Number of cross-validation folds.
        seed: Random seed for deterministic fold assignment.
    """

    name = "binary_activity"
    url = _DATA_URL

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
        input_file_path: str = "output_binary_with_inf_all.csv",
        solvent_list_path: str = "solvent_list.csv",
        data_dir: Optional[str] = None,
        split_mode: str = "all",
        split_part: str = "all",
        fold: int = 0,
        num_folds: int = 5,
        seed: int = 2021,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.solvent_data = {}
        self.build_molecule = BuildMolecule(format="smiles")

        if data_dir is not None:
            input_file_path, solvent_list_path = self._download_data(
                data_dir, input_file_path, solvent_list_path
            )

        self.input_file_path = input_file_path
        self.solvent_list_path = solvent_list_path
        if input_file_path and os.path.exists(input_file_path):
            solvent_list = pd.read_csv(solvent_list_path, index_col="solvent_id")
            self.dataset = pd.read_csv(input_file_path, low_memory=False)
            self._validate_dataset_columns()
            self.dataset = self._apply_split(
                split_mode=split_mode,
                split_part=split_part,
                fold=fold,
                num_folds=num_folds,
                seed=seed,
            )
            self.solvent_smiles = solvent_list["smiles_can"].to_dict()
        else:
            if split_mode != "all" or split_part != "all":
                raise FileNotFoundError(
                    "A split was requested, but the binary-mixture CSV is unavailable."
                )
            self.dataset = pd.DataFrame()
            self.solvent_smiles = {}

        self.split_mode = split_mode
        self.split_part = split_part
        self.fold = fold
        self.num_folds = num_folds
        self.seed = seed
        logger.info(
            f"Load {len(self.dataset)} binary-mixture samples from {input_file_path}"
        )

    def _download_data(self, data_dir, input_file_path, solvent_list_path):
        os.makedirs(data_dir, exist_ok=True)

        input_full_path = os.path.join(data_dir, input_file_path)
        solvent_full_path = os.path.join(data_dir, solvent_list_path)

        if not os.path.exists(input_full_path):
            url = _DATA_URL + input_file_path
            logger.info("Downloading {} to {}".format(url, input_full_path))
            get_path_from_url(url, data_dir, decompress=False)

        if not os.path.exists(solvent_full_path):
            url = _DATA_URL + solvent_list_path
            logger.info("Downloading {} to {}".format(url, solvent_full_path))
            get_path_from_url(url, data_dir, decompress=False)

        return input_full_path, solvent_full_path

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, paddle.Tensor):
            idx = idx.item()

        data = self.dataset.iloc[idx]
        ids = [data["solv1"], data["solv2"]]

        for sid in ids:
            if sid not in self.solvent_data:
                self._generate_solvent(sid)

        solv1 = self.solvent_data[ids[0]]
        solv2 = self.solvent_data[ids[1]]

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
        }
        return sample

    def _generate_solvent(self, solvent_id):
        smiles = self.solvent_smiles[solvent_id]
        mol = self.build_molecule(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for solvent {solvent_id}: {smiles}")

        graph = smiles_to_pgl_graph(smiles, add_self_loop=True)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)

        self.solvent_data[solvent_id] = {
            "graph": graph,
            "hba": hba,
            "hbd": hbd,
            "intra_hb": min(hba, hbd),
        }

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



class BinaryActivityCollator(object):
    """Batch binary activity-coefficient samples for the GE-GNN model."""

    def __init__(self):
        pass

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        batched_g1 = pgl.Graph.batch([sample["g1"] for sample in batch])
        batched_g2 = pgl.Graph.batch([sample["g2"] for sample in batch])
        for graph in (batched_g1, batched_g2):
            for key, value in graph.node_feat.items():
                if not isinstance(value, paddle.Tensor):
                    graph.node_feat[key] = paddle.to_tensor(value, dtype="float32")

        batched_sample = {
            "g1": batched_g1,
            "g2": batched_g2,
            "empty_solvsys": BinaryActivityDataset.generate_solvsys(len(batch)),
        }
        float_keys = (
            "x1",
            "x2",
            "intra_hb1",
            "intra_hb2",
            "inter_hb",
        )
        if all("gamma1" in sample and "gamma2" in sample for sample in batch):
            float_keys += ("gamma1", "gamma2")
        for key in float_keys:
            values = np.asarray([sample[key] for sample in batch], dtype=np.float32)
            batched_sample[key] = paddle.to_tensor(
                values.reshape(-1, 1), dtype="float32"
            )

        if "gamma1" in batched_sample:
            batched_sample["gamma"] = paddle.concat(
                [batched_sample["gamma1"], batched_sample["gamma2"]], axis=1
            )
        return batched_sample
