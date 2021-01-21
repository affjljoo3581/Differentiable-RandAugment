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

import numpy as np
import torch
import torch.optim as optim

from differentiable_randaugment.randaugment import RandAugment, RandAugmentModule


def test_RandAugmentModule_is_differentiable():
    augmentor = RandAugmentModule(num_ops=3, normalized=True)
    optimizer = optim.SGD(augmentor.parameters(), lr=1e-1)

    initial_magnitude = augmentor.get_magnitude()

    # Update `RandAugment` module.
    for _ in range(10):
        img = torch.rand((1, 3, 32, 32), dtype=torch.float, requires_grad=True)
        augmentor(img).sum().backward()
        optimizer.step()

    assert augmentor.get_magnitude() != initial_magnitude


def test_RandAugment_no_exceptions():
    img = np.random.randint(0, 0xFF, (32, 32, 3), dtype=np.uint8)
    augmentor = RandAugment(num_ops=3, magnitude=1.0)

    for _ in range(100):
        augmentor(image=img)
