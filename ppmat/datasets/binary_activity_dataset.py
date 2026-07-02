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

import numpy as np
import paddle
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.utils import download
from ppmat.utils import logger

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
_ALLOWABLE_DEGREES = [0, 1, 2, 3, 4, 5]
_ALLOWABLE_VALENCES = [0, 1, 2, 3, 4, 5, 6]
_ALLOWABLE_CHARGES = [-2, -1, 0, 1, 2]
_ALLOWABLE_RADICAL_ELECTRONS = [0, 1, 2]
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
    feats = (
        _one_hot_encoding(atom.GetSymbol(), _ALLOWABLE_ATOM_TYPES)
        + _one_hot_encoding(atom.GetDegree(), _ALLOWABLE_DEGREES)
        + _one_hot_encoding(atom.GetImplicitValence(), _ALLOWABLE_VALENCES)
        + _one_hot_encoding(atom.GetFormalCharge(), _ALLOWABLE_CHARGES)
        + _one_hot_encoding(atom.GetNumRadicalElectrons(), _ALLOWABLE_RADICAL_ELECTRONS)
        + _one_hot_encoding(atom.GetHybridization(), _ALLOWABLE_HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + _one_hot_encoding(atom.GetTotalNumHs(), _ALLOWABLE_NUM_H)
    )
    return np.array(feats, dtype=np.float32)


_DATA_URL = "https://raw.githubusercontent.com/avt-svt-public/GDI-NN/main/data/"


def smiles_to_pgl_graph(smiles, add_self_loop=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES: %s", smiles)
        return None

    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)

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


def biPxy(x1, solv1_gam, solv2_gam, solv1_psat, solv2_psat):
    solv1_p = x1 * np.exp(solv1_gam) * solv1_psat
    solv2_p = (1 - x1) * np.exp(solv2_gam) * solv2_psat
    equi_p = solv1_p + solv2_p
    y1 = solv1_p / equi_p
    return y1, equi_p


class BinaryActivityDataset(Dataset):
    def __init__(
        self,
        input_file_path="output_binary_with_inf_all.csv",
        solvent_list_path="solvent_list.csv",
        data_dir=None,
        generate_all=False,
        return_hbond=True,
        return_comp=True,
        return_gamma=True,
    ):
        super().__init__()

        self.return_hbond = return_hbond
        self.return_comp = return_comp
        self.return_gamma = return_gamma
        self.solvent_data = {}

        if data_dir is not None:
            input_file_path, solvent_list_path = self._download_data(
                data_dir, input_file_path, solvent_list_path
            )

        if input_file_path and os.path.exists(input_file_path):
            solvent_list = pd.read_csv(solvent_list_path, index_col="solvent_id")
            self.dataset = pd.read_csv(input_file_path)
            self.solvent_names = solvent_list["solvent_name"].to_dict()
            self.solvent_smiles = solvent_list["smiles_can"].to_dict()
        else:
            self.dataset = pd.DataFrame()
            self.solvent_names = {}
            self.solvent_smiles = {}

        if generate_all:
            self._generate_all()

    def _download_data(self, data_dir, input_file_path, solvent_list_path):
        os.makedirs(data_dir, exist_ok=True)

        input_full_path = os.path.join(data_dir, input_file_path)
        solvent_full_path = os.path.join(data_dir, solvent_list_path)

        if not os.path.exists(input_full_path):
            url = _DATA_URL + input_file_path
            logger.info("Downloading {} to {}".format(url, input_full_path))
            download(url, input_full_path)

        if not os.path.exists(solvent_full_path):
            url = _DATA_URL + solvent_list_path
            logger.info("Downloading {} to {}".format(url, solvent_full_path))
            download(url, solvent_full_path)

        return input_full_path, solvent_full_path

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
            "solv1_id": int(ids[0].split("_")[1]),
            "solv2_id": int(ids[1].split("_")[1]),
        }
        return sample

    def _generate_solvent(self, solvent_id):
        smiles = self.solvent_smiles[solvent_id]
        mol = Chem.MolFromSmiles(smiles)

        graph = smiles_to_pgl_graph(smiles, add_self_loop=True)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)

        self.solvent_data[solvent_id] = {
            "graph": graph,
            "hba": hba,
            "hbd": hbd,
            "intra_hb": min(hba, hbd),
        }

    def _generate_all(self):
        for solvent_id in self.solvent_smiles:
            if solvent_id not in self.solvent_data:
                self._generate_solvent(solvent_id)

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

    def generate_sample(self, chemical_list, smiles_list, solv1_x, gamma_list=None):
        solvent_data = {}
        for i, sml in enumerate(smiles_list):
            sml = Chem.MolToSmiles(Chem.MolFromSmiles(sml))
            mol = Chem.MolFromSmiles(sml)
            graph = smiles_to_pgl_graph(sml, add_self_loop=True)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            solvent_data[chemical_list[i]] = {
                "graph": graph,
                "hba": hba,
                "hbd": hbd,
                "intra_hb": min(hba, hbd),
            }

        solv1 = solvent_data[chemical_list[0]]
        solv2 = solvent_data[chemical_list[1]]

        sample = {
            "g1": solv1["graph"],
            "g2": solv2["graph"],
            "intra_hb1": solv1["intra_hb"],
            "intra_hb2": solv2["intra_hb"],
            "inter_hb": min(solv1["hba"], solv2["hbd"])
            + min(solv1["hbd"], solv2["hba"]),
            "x1": solv1_x,
            "x2": 1.0 - solv1_x,
        }
        if gamma_list is not None:
            sample["gamma1"] = gamma_list[0]
            sample["gamma2"] = gamma_list[1]
        return sample

    def search_chemical(self, chemical_name):
        for solvent_id in self.solvent_names:
            if chemical_name.lower() == self.solvent_names[solvent_id].lower():
                return [
                    solvent_id,
                    self.dataset[
                        (self.dataset["solv1"] == solvent_id)
                        | (self.dataset["solv2"] == solvent_id)
                    ].index.to_list(),
                ]
        return None

    def search_chemical_pair(self, chemical_list):
        solv1_match = self.search_chemical(chemical_list[0])[0]
        solv2_match = self.search_chemical(chemical_list[1])[0]
        return [
            [solv1_match, solv2_match],
            self.dataset[
                (
                    (self.dataset["solv1"] == solv1_match)
                    & (self.dataset["solv2"] == solv2_match)
                )
                | (
                    (self.dataset["solv2"] == solv1_match)
                    & (self.dataset["solv1"] == solv2_match)
                )
            ].index.to_list(),
        ]

    def node_to_atom_list(self, idx):
        sample = self.__getitem__(idx)
        g1 = sample["g1"]
        g2 = sample["g2"]
        allowable_set = [
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

        node_feat1 = g1.node_feat["h"]
        if isinstance(node_feat1, paddle.Tensor):
            node_feat1 = node_feat1.numpy()
        node_feat1 = node_feat1[:, : len(allowable_set)]
        atom_list1 = []
        for i in range(g1.num_nodes):
            atom_list1.append(allowable_set[np.where(node_feat1[i] == 1)[0][0]])

        node_feat2 = g2.node_feat["h"]
        if isinstance(node_feat2, paddle.Tensor):
            node_feat2 = node_feat2.numpy()
        node_feat2 = node_feat2[:, : len(allowable_set)]
        atom_list2 = []
        for i in range(g2.num_nodes):
            atom_list2.append(allowable_set[np.where(node_feat2[i] == 1)[0][0]])

        return {"atom_list": [atom_list1, atom_list2]}

    def get_smiles(self, idx):
        return {
            "smiles": [
                self.dataset["solv1_smiles"].iloc[idx],
                self.dataset["solv2_smiles"].iloc[idx],
            ]
        }

    def get_solvx(self, idx):
        return {
            "solv_x": [
                self.dataset["solv1_x"].iloc[idx],
                1.0 - self.dataset["solv1_x"].iloc[idx],
            ]
        }


class BinaryActivityCollator(object):
    def __init__(self, batch_size=5):
        self.batch_size = batch_size

    def __call__(self, batch):
        g1_list = [sample["g1"] for sample in batch]
        g2_list = [sample["g2"] for sample in batch]

        batched_g1 = pgl.Graph.batch(g1_list)
        batched_g2 = pgl.Graph.batch(g2_list)

        for g in [batched_g1, batched_g2]:
            for key in g.node_feat:
                val = g.node_feat[key]
                if not isinstance(val, paddle.Tensor):
                    g.node_feat[key] = paddle.to_tensor(val, dtype="float32")

        batch_size = len(batch)
        empty_solvsys = BinaryActivityDataset.generate_solvsys(batch_size)

        keys = [
            "x1",
            "x2",
            "gamma1",
            "gamma2",
            "intra_hb1",
            "intra_hb2",
            "inter_hb",
            "solv1_id",
            "solv2_id",
        ]

        batched_sample = {
            "g1": batched_g1,
            "g2": batched_g2,
            "empty_solvsys": empty_solvsys,
        }

        for key in keys:
            values = [sample[key] for sample in batch]
            arr = np.array(values, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            batched_sample[key] = paddle.to_tensor(arr, dtype="float32")

        batched_sample["gamma"] = paddle.concat(
            [batched_sample["gamma1"], batched_sample["gamma2"]], axis=1
        )

        return batched_sample
