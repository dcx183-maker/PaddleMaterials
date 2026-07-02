# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import paddle
import paddle.nn as nn


class GibbsDuhemLoss(nn.Layer):
    def __init__(
        self,
        pinn_lambda: float = 1.0,
    ):
        super().__init__()
        if pinn_lambda < 0:
            raise ValueError(
                f"pinn_lambda should be non-negative, but got {pinn_lambda}"
            )
        self.pinn_lambda = pinn_lambda

    def forward(
        self,
        gamma_grad_y1: paddle.Tensor,
        gamma_grad_y2: paddle.Tensor,
        x1: paddle.Tensor,
    ) -> paddle.Tensor:

        gd_residual = x1 * gamma_grad_y1 + (1 - x1) * gamma_grad_y2
        loss = self.pinn_lambda * paddle.mean(gd_residual**2)
        return loss
