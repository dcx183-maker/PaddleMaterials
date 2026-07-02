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

from __future__ import annotations

import paddle


def get_activation(activation, get_nn=False):
    if activation is None or activation in ["relu", "ReLU", "RELU"]:
        if get_nn:
            return paddle.nn.ReLU
        return paddle.nn.functional.relu
    elif activation in ["elu", "ELU"]:
        if get_nn:
            return paddle.nn.ELU
        return paddle.nn.functional.elu
    elif activation in [
        "LeakyReLU",
        "LeakyRELU",
        "leakyReLU",
        "leakyrelu",
        "leakyRELU",
        "leaky_relu",
        "Leaky_ReLU",
        "Leaky_RELU",
    ]:
        if get_nn:
            return paddle.nn.LeakyReLU
        return paddle.nn.functional.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn:
            return paddle.nn.Sigmoid
        return paddle.nn.functional.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn:
            return paddle.nn.Softplus
        return paddle.nn.functional.softplus
    elif activation in ["silu", "SiLU", "SILU"]:
        if get_nn:
            return paddle.nn.SiLU
        return paddle.nn.functional.silu
    if get_nn:
        return paddle.nn.ReLU
    return paddle.nn.functional.relu


def get_mlp_module(dim_in, dim_hidden, dropout):
    mlp_module_list = paddle.nn.LayerList()
    mlp_module_list.append(
        paddle.nn.Sequential(
            paddle.nn.Embedding(dim_in, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(dropout),
            paddle.nn.Linear(dim_hidden, dim_hidden),
            paddle.nn.ReLU(),
        )
    )
    return mlp_module_list


class MCMMultiMLP(paddle.nn.Layer):
    def __init__(
        self,
        solvent_id_max,
        dim_hidden_channels=128,
        dropout_hidden=0.05,
        dropout_interaction=0.03,
        mlp_activation="relu",
        mlp_num_hid_layers=1,
        pinn_lambda=1.0,
        property_name="gamma",
        **kwargs,
    ):
        super().__init__()
        self.mlp_activation = get_activation(mlp_activation, get_nn=True)
        self.dropout_p1 = dropout_hidden
        self.dropout_p2 = dropout_interaction
        self.dim_hidden_channels = dim_hidden_channels
        self.pinn_lambda = pinn_lambda
        self.property_name = property_name
        if isinstance(self.property_name, (list, tuple)):
            self.property_name = self.property_name[0]
        self.solvent_emb = get_mlp_module(
            solvent_id_max + 1, self.dim_hidden_channels, self.dropout_p1
        )
        mid_emb = 2 * self.dim_hidden_channels
        list_layers_end_1 = [
            paddle.nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation(),
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_1.append(paddle.nn.Linear(mid_emb, mid_emb))
                list_layers_end_1.append(self.mlp_activation())
        list_layers_end_1.append(paddle.nn.Linear(mid_emb, 1))
        list_layers_end_2 = [
            paddle.nn.Linear(mid_emb + 2, mid_emb),
            self.mlp_activation(),
        ]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_2.append(paddle.nn.Linear(mid_emb, mid_emb))
                list_layers_end_2.append(self.mlp_activation())
        list_layers_end_2.append(paddle.nn.Linear(mid_emb, 1))
        self.layers_end = paddle.nn.LayerList(
            [
                paddle.nn.Sequential(*list_layers_end_1),
                paddle.nn.Sequential(*list_layers_end_2),
            ]
        )

    def _forward(self, data):
        solv1x = data["x1"]
        if not isinstance(solv1x, paddle.Tensor):
            solv1x = paddle.to_tensor(solv1x, dtype="float32")
        if solv1x.ndim == 1:
            solv1x = solv1x.unsqueeze(1)
        solv1_id = data["solv1_id"]
        solv2_id = data["solv2_id"]

        x_solvent = self.solvent_emb[0](solv1_id)
        x_solute = self.solvent_emb[0](solv2_id)
        h = paddle.concat([x_solvent, solv1x, x_solute, 1 - solv1x], axis=1).astype(
            "float32"
        )
        output_y1 = self.layers_end[0](h)
        output_y2 = self.layers_end[1](h)
        output = paddle.concat([output_y1, output_y2], axis=1)
        return output

    def forward(self, data, return_loss=True, return_prediction=True):
        output = self._forward(data)

        loss_dict = {}
        if return_loss:
            gamma1_label = data["gamma1"]
            gamma2_label = data["gamma2"]
            if not isinstance(gamma1_label, paddle.Tensor):
                gamma1_label = paddle.to_tensor(gamma1_label, dtype="float32")
            if not isinstance(gamma2_label, paddle.Tensor):
                gamma2_label = paddle.to_tensor(gamma2_label, dtype="float32")
            if gamma1_label.ndim == 1:
                gamma1_label = gamma1_label.unsqueeze(1)
            if gamma2_label.ndim == 1:
                gamma2_label = gamma2_label.unsqueeze(1)
            pred_loss = paddle.nn.functional.mse_loss(
                output, paddle.concat([gamma1_label, gamma2_label], axis=1)
            )
            loss_dict["loss"] = pred_loss

        prediction = {}
        if return_prediction:
            prediction[self.property_name] = output

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, data):
        output = self._forward(data)
        return {self.property_name: output}
