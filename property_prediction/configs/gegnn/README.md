# GE-GNN

[Thermodynamics-consistent graph neural networks](https://doi.org/10.1039/D4SC04554H)

## Abstract

GE-GNN (Excess Gibbs Free Energy Graph Neural Network) predicts the dimensionless excess Gibbs free energy $G^E$ of binary solvent mixtures. The logarithmic activity coefficients are obtained by differentiating the predicted free energy with respect to the mole fraction $x_1$, which satisfies the Gibbs–Duhem relation by construction:

```math
\ln \gamma_1 = G^E + (1 - x_1) \frac{dG^E}{dx_1}
```

```math
\ln \gamma_2 = G^E - x_1 \frac{dG^E}{dx_1}
```

## Datasets

GE-GNN uses binary-mixture activity-coefficient data and solvent metadata.

| Dataset | Files | Download |
| --- | --- | --- |
| binaryGamma | `output_binary_with_inf_all.csv`, `solvent_list.csv` | [binaryGamma](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/thermodynamic_data_of_binary_mixtures/) |

## Model

GE-GNN first encodes each solvent molecule with two GCN convolution layers, then aggregates molecular-level features through an MPNN interaction graph. The excess Gibbs free energy is predicted by an MLP head, and the activity coefficients are derived via analytic differentiation.

## Results

| Model Name | Dataset | Target | MAE (Val) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| gegnn_binary_activity | binaryGamma | $\ln \gamma_1$, $\ln \gamma_2$ | Not evaluated | [gegnn_binary_activity.yaml](gegnn_binary_activity.yaml) | [gegnn_binary_activity](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/ge-gnn/gegnn_binary_activity.zip) |

The supplied configuration trains on the complete CSV and selects the best
checkpoint using the training metric. It defines no validation split.
The interaction batching has been corrected to keep each mixture independent;
previously trained weights need reevaluation and retraining before reporting
reproduction metrics. A training
metric or a same-framework determinism check does not establish the MIIT
cross-framework accuracy requirements.

The vocabulary is included in the YAML so training and prediction use the same
74 atom features. `build_vocab` resolves its categorical tokens;
`MolecularGraphConverter` encodes atoms under `node_feat["h"]`. `BuildSolvent`
builds the molecule, graph and hydrogen-bond descriptors, and
`BuildBinaryMixture` encodes the mixture. The dataset stores and loads those
built objects using the indexed graph cache. Changing the vocabulary, molecule
options, graph options or solvent metadata rebuilds that cache.

Both CSV files are downloaded automatically if absent, with MD5 verification.
For local data, place `solvent_list.csv` beside the activity CSV or set
`solvent_list_path` in the dataset configuration. The activity CSV requires
`solv1`, `solv2`, `solv1_x`, `solv1_gamma` and `solv2_gamma`; the last two columns
contain **logarithmic** activity coefficients. Solvent metadata requires
`solvent_id` and `smiles_can`.

## Training

```bash
python property_prediction/train.py \
    -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml
```

To reproduce a paper split, prepare separate training and held-out CSV files
using the original split protocol, and set the training `path` accordingly.
The trainer adds a timestamp and seed to `Trainer.output_dir`; use the actual
run directory when loading `checkpoints/best.pdparams` or `latest.pdparams`.

## Validation

Evaluation requires an explicit held-out dataset. For example, after preparing
`val.csv` and its solvent metadata:

```bash
python property_prediction/train.py \
    -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Global.do_test=False \
    Dataset.val.dataset.__class_name__=BinaryActivityDataset \
    'Dataset.val.dataset.__init_params__.vocab=${Vocabulary}' \
    'Dataset.val.dataset.__init_params__.build_graph_cfg=${Predict.graph_converter}' \
    Dataset.val.loader.num_workers=0 \
    Dataset.val.loader.use_shared_memory=False \
    Dataset.val.sampler.__class_name__=BatchSampler \
    Dataset.val.sampler.__init_params__.batch_size=256 \
    Dataset.val.dataset.__init_params__.path=./data/binary_activity/val.csv \
    Dataset.val.sampler.__init_params__.shuffle=False \
    Dataset.val.sampler.__init_params__.drop_last=False \
    Trainer.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/ge-gnn/gegnn_binary_activity.zip \
    Trainer.pretrained_weight_name=best.pdparams
```

Keep `Trainer.eval_with_no_grad=False` and `Predict.eval_with_no_grad=False`
because the activity coefficients require the composition derivative.

## Prediction

Use the current YAML with the downloaded `best.pdparams` checkpoint. The current
YAML includes the vocabulary and preprocessing configuration required by the
model; an older checkpoint package may contain a YAML without those settings.

```python
from ppmat.datasets.build_solvent import BuildSolvent
from ppmat.predictor import PropertyPredictor

predictor = PropertyPredictor(
    config_path="property_prediction/configs/gegnn/gegnn_binary_activity.yaml",
    checkpoint_path="./gegnn_binary_activity/checkpoints/best.pdparams",
)
build_solvent = BuildSolvent(predictor.graph_converter_fn)
result = predictor.from_mixture(build_solvent("CCO"), build_solvent("O"), x1=0.5)
print(result["gamma"])  # ln(gamma1), ln(gamma2)
```

## Verification

```bash
python -m unittest discover -s test -p test_gegnn.py -v
```

The tests cover reference atom features, solvent and mixture preprocessing,
cache reuse and invalidation, forward prediction and two optimizer steps.
Full PyTorch/Paddle accuracy alignment and compiler speed comparisons require
the original model, matching data splits and the supported training environment.

## Citation

```bibtex
@article{rittig2024thermodynamics,
  title={Thermodynamics-consistent graph neural networks},
  author={Rittig, Jan G. and Mitsos, Alexander},
  journal={Chemical Science},
  year={2024},
  doi={10.1039/D4SC04554H}
}
```
