import numpy as np
import paddle


class ManualGCN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, src_nodes, dst_nodes):
        h = paddle.matmul(x, weight)

        in_deg = {}
        out_deg = {}
        for src, dst in zip(src_nodes, dst_nodes):
            in_deg[dst] = in_deg.get(dst, 0) + 1
            out_deg[src] = out_deg.get(src, 0) + 1

        outputs = []
        for v in range(h.shape[0]):
            msgs = []
            for i, (u, dst) in enumerate(zip(src_nodes, dst_nodes)):
                if dst == v:
                    scale = 1.0 / (
                        (out_deg.get(u, 1) ** 0.5) * (in_deg.get(v, 1) ** 0.5)
                    )
                    msgs.append(h[u] * scale)
            outputs.append(
                paddle.stack(msgs, axis=0).sum(axis=0)
                if msgs
                else paddle.zeros([h.shape[1]], dtype=h.dtype)
            )
        return paddle.stack(outputs, axis=0)


class ManualNNConv(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, node_feats, edge_weights, src_nodes, dst_nodes):
        outputs = []
        for v in range(node_feats.shape[0]):
            msgs_v = [
                paddle.matmul(node_feats[src].unsqueeze(0), edge_weights[i]).squeeze(0)
                for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes))
                if dst == v
            ]
            outputs.append(
                paddle.stack(msgs_v, axis=0).sum(axis=0)
                if msgs_v
                else paddle.zeros([node_feats.shape[1]], dtype=node_feats.dtype)
            )
        return paddle.stack(outputs, axis=0)


class SolvGNNPaddle(paddle.nn.Layer):
    def __init__(self, in_dim=64, hidden_dim=64, n_classes=2):
        super().__init__()
        self.conv1 = ManualGCN()
        self.conv2 = ManualGCN()
        self.nnconv = ManualNNConv()

        self.w_conv1 = self.create_parameter(
            shape=[in_dim, hidden_dim], dtype="float32"
        )
        self.w_conv2 = self.create_parameter(
            shape=[hidden_dim, hidden_dim], dtype="float32"
        )

        self.w_proj = self.create_parameter(
            shape=[hidden_dim + 1, hidden_dim], dtype="float32"
        )
        self.b_proj = self.create_parameter(shape=[hidden_dim], dtype="float32")

        self.w_ih = self.create_parameter(
            shape=[hidden_dim, hidden_dim * 3], dtype="float32"
        )
        self.w_hh = self.create_parameter(
            shape=[hidden_dim, hidden_dim * 3], dtype="float32"
        )
        self.b_ih = self.create_parameter(shape=[hidden_dim * 3], dtype="float32")
        self.b_hh = self.create_parameter(shape=[hidden_dim * 3], dtype="float32")

        self.w_c1 = self.create_parameter(
            shape=[hidden_dim, hidden_dim], dtype="float32"
        )
        self.b_c1 = self.create_parameter(shape=[hidden_dim], dtype="float32")
        self.w_c2 = self.create_parameter(
            shape=[hidden_dim, hidden_dim], dtype="float32"
        )
        self.b_c2 = self.create_parameter(shape=[hidden_dim], dtype="float32")
        self.w_c3 = self.create_parameter(
            shape=[hidden_dim, n_classes], dtype="float32"
        )
        self.b_c3 = self.create_parameter(shape=[n_classes], dtype="float32")

        self.w_edge0 = self.create_parameter(shape=[1, 32], dtype="float32")
        self.b_edge0 = self.create_parameter(shape=[32], dtype="float32")
        self.w_edge2 = self.create_parameter(
            shape=[32, hidden_dim * hidden_dim], dtype="float32"
        )
        self.b_edge2 = self.create_parameter(
            shape=[hidden_dim * hidden_dim], dtype="float32"
        )

        self.b_nnconv = self.create_parameter(shape=[hidden_dim], dtype="float32")

    def forward(
        self,
        feat,
        solv1_x_val,
        inter_hb_val,
        intra_hb1_val,
        intra_hb2_val,
        edges_u,
        edges_v,
        sys_u,
        sys_v,
        num_nodes=5,
    ):
        h1 = paddle.to_tensor(feat)
        h2 = paddle.to_tensor(feat)
        h1_temp = paddle.nn.functional.relu(
            self.conv1(h1, self.w_conv1, edges_u, edges_v)
        )
        h1_temp = paddle.nn.functional.relu(
            self.conv2(h1_temp, self.w_conv2, edges_u, edges_v)
        )
        h2_temp = paddle.nn.functional.relu(
            self.conv1(h2, self.w_conv1, edges_u, edges_v)
        )
        h2_temp = paddle.nn.functional.relu(
            self.conv2(h2_temp, self.w_conv2, edges_u, edges_v)
        )

        hg1 = h1_temp.mean(axis=0, keepdim=True)
        hg2 = h2_temp.mean(axis=0, keepdim=True)
        solv1x_p = paddle.to_tensor(solv1_x_val)
        hg1 = paddle.concat([hg1, solv1x_p[:, None]], axis=1)
        hg2 = paddle.concat([hg2, 1 - solv1x_p[:, None]], axis=1)
        global_in = paddle.concat([hg1, hg2], axis=0)

        node_feats = paddle.matmul(global_in, self.w_proj) + self.b_proj
        node_feats = paddle.nn.functional.relu(node_feats)

        edge_feats = paddle.to_tensor(
            np.array(
                [inter_hb_val, inter_hb_val, intra_hb1_val, intra_hb2_val],
                dtype=np.float32,
            ).reshape(4, 1)
        )

        e_out = paddle.matmul(edge_feats, self.w_edge0) + self.b_edge0
        e_out = paddle.nn.functional.relu(e_out)
        e_out = paddle.matmul(e_out, self.w_edge2) + self.b_edge2
        edge_weights = paddle.reshape(
            e_out, [-1, self.w_conv2.shape[1], self.w_conv2.shape[1]]
        )

        h_gru = paddle.clone(node_feats).unsqueeze(0)
        hidden_dim = self.w_conv2.shape[1]
        for _ in range(1):
            msg_out = self.nnconv(node_feats, edge_weights, sys_u, sys_v)
            msg_out = msg_out + self.b_nnconv
            node_feats = paddle.nn.functional.relu(msg_out)
            gates_x = paddle.matmul(node_feats, self.w_ih) + self.b_ih
            gates_h = paddle.matmul(h_gru.squeeze(0), self.w_hh) + self.b_hh
            x_r, x_z, x_n = (
                gates_x[:, :hidden_dim],
                gates_x[:, hidden_dim : 2 * hidden_dim],
                gates_x[:, 2 * hidden_dim :],
            )
            h_r, h_z, h_n = (
                gates_h[:, :hidden_dim],
                gates_h[:, hidden_dim : 2 * hidden_dim],
                gates_h[:, 2 * hidden_dim :],
            )
            r = paddle.nn.functional.sigmoid(x_r + h_r)
            z = paddle.nn.functional.sigmoid(x_z + h_z)
            n = paddle.tanh(x_n + r * h_n)
            h_gru = ((1 - z) * n + z * h_gru.squeeze(0)).unsqueeze(0)
            node_feats = paddle.clone(h_gru.squeeze(0))

        out1 = paddle.matmul(node_feats, self.w_c1) + self.b_c1
        out1 = paddle.nn.functional.relu(out1)
        out2 = paddle.matmul(out1, self.w_c2) + self.b_c2
        out2 = paddle.nn.functional.relu(out2)
        output = paddle.matmul(out2, self.w_c3) + self.b_c3

        return paddle.concat([output[:1, :], output[1:2, :]], axis=1)
