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
import pgl
from pgl.nn import GCNConv as GraphConv


class NNConv(paddle.nn.Layer):
    def __init__(self, in_size, out_size, edge_network, aggregator="sum"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.edge_network = edge_network
        self.aggregator = aggregator

    def forward(self, graph, node_feats, edge_feats):
        edges = paddle.to_tensor(graph.edges, dtype="int64")
        src = edges[:, 0]
        dst = edges[:, 1]

        edge_weight = self.edge_network(edge_feats)
        edge_weight = paddle.reshape(edge_weight, [-1, self.out_size, self.in_size])

        h_src = paddle.gather(node_feats, src, axis=0).unsqueeze(-1)
        msg = paddle.matmul(edge_weight, h_src).squeeze(-1)

        sort_idx = paddle.argsort(dst)
        dst_sorted = paddle.gather(dst, sort_idx, axis=0)
        msg_sorted = paddle.gather(msg, sort_idx, axis=0)

        if self.aggregator == "sum":
            out = pgl.math.segment_sum(msg_sorted, dst_sorted)
        elif self.aggregator == "mean":
            out = pgl.math.segment_mean(msg_sorted, dst_sorted)
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")
        return out


def get_activation(activation, get_nn=False):
    if activation is None or activation in ["relu", "ReLU", "RELU"]:
        if get_nn:
            return paddle.nn.ReLU
        return paddle.nn.functional.relu
    elif activation in ["elu", "ELU"]:
        if get_nn:
            return paddle.nn.ELU
        return paddle.nn.functional.elu
    elif activation in ["LeakyReLU", "leakyrelu"]:
        if get_nn:
            return paddle.nn.LeakyReLU
        return paddle.nn.functional.leaky_relu
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn:
            return paddle.nn.Softplus
        return paddle.nn.functional.softplus
    if get_nn:
        return paddle.nn.ReLU
    return paddle.nn.functional.relu


class MPNNconv(paddle.nn.Layer):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats=128,
        edge_hidden_feats=32,
        num_step_message_passing=6,
        activation="relu",
    ):
        super().__init__()
        self.mpnn_activation = get_activation(activation)
        self.project_node_feats = paddle.nn.Sequential(
            paddle.nn.Linear(node_in_feats, node_out_feats),
            get_activation(activation, get_nn=True)(),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = paddle.nn.Sequential(
            paddle.nn.Linear(edge_in_feats, edge_hidden_feats),
            get_activation(activation, get_nn=True)(),
            paddle.nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConv(
            in_size=node_out_feats,
            out_size=node_out_feats,
            edge_network=edge_network,
            aggregator="sum",
        )
        self.gru = paddle.nn.GRU(
            input_size=node_out_feats,
            hidden_size=node_out_feats,
            time_major=True,
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)
        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class SolvGNNBinary(paddle.nn.Layer):
    @staticmethod
    def _pgl_graph_to_tensor(graph):
        if not hasattr(graph, "_tensor_mode") or not graph._tensor_mode:
            graph = graph.tensor()
        return graph

    def __init__(
        self,
        in_dim=74,
        hidden_dim=256,
        n_classes=1,
        mlp_dropout_rate=0.0,
        mlp_activation="relu",
        mpnn_activation="relu",
        mlp_num_hid_layers=2,
        pinn_lambda=1.0,
        property_name="gamma",
    ):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(
            node_in_feats=hidden_dim + 1,
            edge_in_feats=1,
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=1,
            activation=mpnn_activation,
        )
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = paddle.nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = paddle.nn.Linear(hidden_dim, n_classes)
        self.pinn_lambda = pinn_lambda
        if isinstance(property_name, (list, tuple)):
            property_name = property_name[0]
        self.property_name = property_name

    def _forward(self, data):
        g1 = data["g1"]
        g2 = data["g2"]

        h1 = g1.node_feat["h"]
        h2 = g2.node_feat["h"]
        if not isinstance(h1, paddle.Tensor):
            h1 = paddle.to_tensor(h1, dtype="float32")
        if not isinstance(h2, paddle.Tensor):
            h2 = paddle.to_tensor(h2, dtype="float32")

        g1 = self._pgl_graph_to_tensor(g1)
        g2 = self._pgl_graph_to_tensor(g2)
        solv1x = paddle.cast(data["x1"], "float32")
        if solv1x.ndim == 1:
            solv1x = solv1x.unsqueeze(1)
        inter_hb = paddle.cast(data["inter_hb"], "float32")
        if inter_hb.ndim == 1:
            inter_hb = inter_hb.unsqueeze(1)
        intra_hb1 = paddle.cast(data["intra_hb1"], "float32")
        if intra_hb1.ndim == 1:
            intra_hb1 = intra_hb1.unsqueeze(1)
        intra_hb2 = paddle.cast(data["intra_hb2"], "float32")
        if intra_hb2.ndim == 1:
            intra_hb2 = intra_hb2.unsqueeze(1)
        empty_solvsys = data["empty_solvsys"]

        h1_temp = paddle.nn.functional.relu(self.conv1(g1, h1))
        h1_temp = paddle.nn.functional.relu(self.conv2(g1, h1_temp))
        h2_temp = paddle.nn.functional.relu(self.conv1(g2, h2))
        h2_temp = paddle.nn.functional.relu(self.conv2(g2, h2_temp))

        hg1 = pgl.math.segment_mean(h1_temp, g1.graph_node_id)
        hg2 = pgl.math.segment_mean(h2_temp, g2.graph_node_id)

        hg1 = paddle.concat((hg1, solv1x), axis=1)
        hg2 = paddle.concat((hg2, 1 - solv1x), axis=1)

        inter_hb_2x = inter_hb.tile([2, 1])
        edge_feats = paddle.concat([inter_hb_2x, intra_hb1, intra_hb2], axis=0)
        hg = self.global_conv1(
            empty_solvsys, paddle.concat((hg1, hg2), axis=0), edge_feats
        )

        output = self.mlp_activation(self.classify1(hg))
        output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)

        output = paddle.concat(
            (output[0 : len(output) // 2, :], output[len(output) // 2 :, :]),
            axis=1,
        )
        return output

    def forward(self, data, return_loss=True, return_prediction=True):
        output = self._forward(data)

        loss_dict = {}
        if return_loss:
            gamma1_label = paddle.cast(data["gamma1"], "float32")
            gamma2_label = paddle.cast(data["gamma2"], "float32")
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


class SolvGNNxMLPBinary(SolvGNNBinary):

    pass


class GEGNNBinary(SolvGNNBinary):

    pass
