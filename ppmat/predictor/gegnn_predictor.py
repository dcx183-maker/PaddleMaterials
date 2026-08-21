# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional
from typing import Sequence

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.gegnn_dataset import _MOLECULAR_GRAPH_CFG
from ppmat.datasets.gegnn_dataset import build_molecular_graph
from ppmat.models import build_graph_converter
from ppmat.predictor.base import BasePredictor


class GEGNNPredictor(BasePredictor):
    """Predict binary-mixture activity coefficients with GE-GNN."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            config_overrides=config_overrides,
        )
        self.load_inference_model()

    def from_binary_mixture(self, smiles1: str, smiles2: str, x1: float):
        """Predict both logarithmic activity coefficients for two SMILES strings."""
        x1 = float(x1)
        if not 0.0 <= x1 <= 1.0:
            raise ValueError("x1 must be in the interval [0, 1].")

        molecules = [BuildMolecule(format="smiles")(smiles) for smiles in (smiles1, smiles2)]
        if any(molecule is None for molecule in molecules):
            raise ValueError("Both mixture components must be valid SMILES strings.")
        converter = build_graph_converter(_MOLECULAR_GRAPH_CFG)
        graphs = [
            build_molecular_graph(molecule, converter) for molecule in molecules
        ]
        if any(molecule is None for molecule in molecules) or any(
            graph is None for graph in graphs
        ):
            raise ValueError("Both mixture components must be valid SMILES strings.")

        from rdkit.Chem import rdMolDescriptors

        hba = [rdMolDescriptors.CalcNumHBA(molecule) for molecule in molecules]
        hbd = [rdMolDescriptors.CalcNumHBD(molecule) for molecule in molecules]
        sample = {
            "g1": graphs[0],
            "g2": graphs[1],
            "x1": x1,
            "x2": 1.0 - x1,
            "intra_hb1": min(hba[0], hbd[0]),
            "intra_hb2": min(hba[1], hbd[1]),
            "inter_hb": min(hba[0], hbd[1]) + min(hbd[0], hba[1]),
            "empty_solvsys": BinaryActivityDataset.generate_solvsys(1),
        }
        return self._run_model(DefaultCollator()([sample]))
