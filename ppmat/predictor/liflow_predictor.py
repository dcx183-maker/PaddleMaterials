# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ppmat.predictor.base import BasePredictor
from ppmat.predictor.gegnn_predictor import GEGNNPredictor


class LiFlowPredictor(BasePredictor):
    """Predictor entry point for LiFlow models using the standard base API."""

    from_binary_mixture = GEGNNPredictor.from_binary_mixture
