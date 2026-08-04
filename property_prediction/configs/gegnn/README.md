# GE-GNN

[Graph Neural Networks with Thermodynamic Insights for Binary Activity Coefficient Prediction](https://doi.org/10.1039/D3DD00112A)

## Abstract

GE-GNN (Excess Gibbs Free Energy Graph Neural Network) predicts the dimensionless excess Gibbs free energy $G^E$ of binary solvent mixtures. The logarithmic activity coefficients are obtained by differentiating the predicted free energy with respect to the mole fraction $x_1$:

```math
\ln \gamma_1 = G^E + (1 - x_1) \frac{dG^E}{dx_1}
```

```math
\ln \gamma_2 = G^E - x_1 \frac{dG^E}{dx_1}
```

The implementation uses the common PaddleMaterials training workflow. The model is implemented in `ppmat/models/gegnn.py`, and the dataset uses the shared `BuildMolecule` factory in `ppmat/datasets/gegnn_dataset.py`.

## Dataset

GE-GNN uses binary-mixture activity-coefficient data and solvent metadata:

| Dataset | Files | Download |
| --- | --- | --- |
| binaryGamma | `output_binary_with_inf_all.csv`, `solvent_list.csv` | [PaddleMaterials dataset storage](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/thermodynamic_data_of_binary_mixtures/) |

The configured `BinaryActivityDataset` automatically downloads missing files to `PPMAT_DATA_DIR` (default: `./ppmat/datasets/gegnn_data`). To use an existing local copy, set:

```bash
export PPMAT_DATA_DIR=./ppmat/datasets/gegnn_data
```

The default configuration uses the upstream `comp_inter` protocol: 5-fold TPSA-stratified cross validation with seed `2021`. Use the same `fold` value from `0` to `4` for both the training and validation datasets.

## Results

| Model | Dataset | Target | Config | Checkpoint |
| --- | --- | --- | --- | --- |
| GE-GNN | binaryGamma | $\ln \gamma_1$, $\ln \gamma_2$ | [gegnn_binary_activity](gegnn_binary_activity.yaml) | [AI Studio model space](https://aistudio.baidu.com/modelsdetail/49433?modelId=49433) |

The model space provides `gegnn_binary_gamma_fold0_best.pdparams` through `gegnn_binary_gamma_fold4_best.pdparams`. Download the file matching the configured validation fold.

## Training

```bash
python property_prediction/train.py \
  -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml
```

## Validation

```bash
python property_prediction/train.py \
  -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml \
  Global.do_train=False \
  Global.do_eval=True \
  Global.do_test=False \
  Trainer.pretrained_model_path=./output/gegnn_binary_gamma/checkpoints/gegnn_binary_gamma_fold0_best.pdparams \
  Dataset.train.dataset.__init_params__.fold=0 \
  Dataset.val.dataset.__init_params__.fold=0
```

Keep `Trainer.eval_with_no_grad=False` because GE-GNN needs $dG^E/dx_1$ to calculate the activity coefficients.

## Prediction

```bash
python property_prediction/predict.py \
  --config_path=property_prediction/configs/gegnn/gegnn_binary_activity.yaml \
  --checkpoint_path=./output/gegnn_binary_gamma/checkpoints/gegnn_binary_gamma_fold0_best.pdparams \
  --smiles1=CCO \
  --smiles2=O \
  --x1=0.5
```

`smiles1` and `smiles2` are the two solvent components; `x1` is the mole fraction of `smiles1` and must be in the interval $[0, 1]$.

## Verification

```bash
python -m pytest -q test/test_gegnn.py
```

## Citation

```bibtex
@article{hasse2023graph,
  title={Graph Neural Networks with Thermodynamic Insights for Binary Activity Coefficient Prediction},
  author={Hasse, Florian and others},
  journal={Digital Discovery},
  year={2023}
}
```

## References

- [PaddleMaterials issue #258](https://github.com/PaddlePaddle/PaddleMaterials/issues/258)
- [Upstream GDI-NN implementation](https://git.rwth-aachen.de/avt-svt/public/GDI-NN)
