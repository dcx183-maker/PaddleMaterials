import sys
import os
import unittest
import numpy as np

import paddle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


class TestGDINNModels(unittest.TestCase):

    def test_solvgnn_instantiation(self):
        from ppmat.models.gdinn.model_gnn import SolvGNNBinary
        model = SolvGNNBinary(in_dim=74, hidden_dim=32, n_classes=1)
        self.assertIsNotNone(model)

    def test_solvgnn_xmlp_instantiation(self):
        from ppmat.models.gdinn.model_gnn import SolvGNNxMLPBinary
        model = SolvGNNxMLPBinary(in_dim=74, hidden_dim=32, n_classes=1)
        self.assertIsNotNone(model)

    def test_gegnn_instantiation(self):
        from ppmat.models.gdinn.model_gnn import GEGNNBinary
        model = GEGNNBinary(in_dim=74, hidden_dim=32, n_classes=1)
        self.assertIsNotNone(model)

    def test_mcm_instantiation(self):
        from ppmat.models.gdinn.model_mcm import MCMMultiMLP
        model = MCMMultiMLP(solvent_id_max=10, dim_hidden_channels=32)
        self.assertIsNotNone(model)

    def test_mcm_forward(self):
        from ppmat.models.gdinn.model_mcm import MCMMultiMLP
        model = MCMMultiMLP(solvent_id_max=10, dim_hidden_channels=32)
        data = {
            "x1": paddle.rand([4, 1]),
            "solv1_id": paddle.randint(0, 10, [4]),
            "solv2_id": paddle.randint(0, 10, [4]),
            "gamma1": paddle.randn([4, 1]),
            "gamma2": paddle.randn([4, 1]),
        }
        out = model(data)
        self.assertIn("loss_dict", out)
        self.assertIn("pred_dict", out)
        self.assertEqual(out["pred_dict"]["gamma"].shape, [4, 2])

    def test_mcm_predict(self):
        from ppmat.models.gdinn.model_mcm import MCMMultiMLP
        model = MCMMultiMLP(solvent_id_max=10, dim_hidden_channels=32)
        data = {
            "x1": paddle.rand([4, 1]),
            "solv1_id": paddle.randint(0, 10, [4]),
            "solv2_id": paddle.randint(0, 10, [4]),
        }
        out = model.predict(data)
        self.assertIn("gamma", out)
        self.assertEqual(out["gamma"].shape, [4, 2])


class TestGibbsDuhemLoss(unittest.TestCase):

    def test_loss_instantiation(self):
        from ppmat.losses.gibbs_duhem_loss import GibbsDuhemLoss
        loss_fn = GibbsDuhemLoss(pinn_lambda=1.0)
        self.assertIsNotNone(loss_fn)

    def test_loss_forward(self):
        from ppmat.losses.gibbs_duhem_loss import GibbsDuhemLoss
        loss_fn = GibbsDuhemLoss(pinn_lambda=1.0)
        grad1 = paddle.randn([4, 1])
        grad2 = paddle.randn([4, 1])
        x1 = paddle.rand([4, 1])
        loss = loss_fn(grad1, grad2, x1)
        self.assertEqual(loss.shape, [])
        self.assertGreater(float(loss), 0)

    def test_loss_zero_residual(self):
        from ppmat.losses.gibbs_duhem_loss import GibbsDuhemLoss
        loss_fn = GibbsDuhemLoss(pinn_lambda=1.0)
        x1 = paddle.to_tensor([[0.5]])
        grad1 = paddle.to_tensor([[-1.0]])
        grad2 = paddle.to_tensor([[1.0]])
        loss = loss_fn(grad1, grad2, x1)
        self.assertAlmostEqual(float(loss), 0.0, places=5)


class TestAtomFeaturization(unittest.TestCase):

    def test_canonical_atom_feats(self):
        from rdkit import Chem
        sys.path.insert(0, os.path.join(BASE_DIR, "ppmat", "datasets"))
        from binary_activity_dataset import _canonical_atom_feats

        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        feats = _canonical_atom_feats(atom)
        self.assertEqual(feats.shape[0], 74)
        self.assertEqual(feats.dtype, np.float32)

    def test_smiles_to_pgl_graph(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "ppmat", "datasets"))
        from binary_activity_dataset import smiles_to_pgl_graph

        graph = smiles_to_pgl_graph("CCO")
        self.assertIsNotNone(graph)
        self.assertEqual(graph.num_nodes, 3)
        self.assertIn("h", graph.node_feat)
        self.assertEqual(graph.node_feat["h"].shape[1], 74)


if __name__ == "__main__":
    unittest.main()
