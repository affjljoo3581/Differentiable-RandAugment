# Copyright 2021 Jungwoo Park
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

import random
from typing import Tuple

import numpy as np
import torch

from differentiable_randaugment.base import BaseOperation


def assert_enhance_op(
    op: BaseOperation,
    num_images: int = 5,
    num_assertions: int = 10,
    image_size: Tuple[int, int] = (64, 64),
):
    for _ in range(num_images):
        # Create new random image.
        img_numpy = np.random.randint(0, 0xFF, image_size + (3,), dtype=np.uint8)
        img_tensor = (torch.from_numpy(img_numpy) / 0xFF).permute(2, 0, 1).unsqueeze(0)

        for _ in range(num_assertions):
            seed = random.random()
            magnitude = random.random()

            # Fix the random seed and apply to the numpy array.
            random.seed(seed)
            output_numpy = op(img_numpy, magnitude)
            output_numpy = output_numpy / 0xFF

            # Fix the random seed and apply to the torch tensor.
            random.seed(seed)
            output_tensor = op(img_tensor, torch.tensor(magnitude))
            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)

            assert np.isclose(output_numpy, output_tensor.numpy(), atol=5 / 0xFF).all()


def assert_geometric_op(
    op: BaseOperation,
    num_images: int = 5,
    num_assertions: int = 10,
    image_size: Tuple[int, int] = (64, 64),
):
    for _ in range(num_images):
        # Create new white image.
        img_numpy = 0xFF * np.ones(image_size + (3,), dtype=np.uint8)
        img_tensor = (torch.from_numpy(img_numpy) / 0xFF).permute(2, 0, 1).unsqueeze(0)

        for _ in range(num_assertions):
            seed = random.random()
            magnitude = random.random()

            # Fix the random seed and apply to the numpy array.
            random.seed(seed)
            output_numpy = op(img_numpy, magnitude)
            output_numpy = output_numpy / 0xFF

            # Fix the random seed and apply to the torch tensor.
            random.seed(seed)
            output_tensor = op(img_tensor, torch.tensor(magnitude))
            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)

            assert np.abs(output_numpy - output_tensor.numpy()).mean() < 0.01
