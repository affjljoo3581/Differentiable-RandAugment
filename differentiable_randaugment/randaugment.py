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
from typing import Any, List

import albumentations as A
import numpy as np
import torch
import torch.nn as nn

from differentiable_randaugment.base import BaseOperation, Identity
from differentiable_randaugment.enhance import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    Sharpness,
    Solarize,
)
from differentiable_randaugment.geometric import (
    Rotate,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
)

DefaultOpSet = [
    Identity(),
    ShearX(0, 0.3),
    ShearY(0, 0.3),
    TranslateX(0, 0.45),
    TranslateY(0, 0.45),
    Rotate(0, 30),
    AutoContrast(),
    Equalize(),
    Solarize(0, 0xFF),
    Posterize(4, 8),
    Contrast(0.1, 1.9),
    Color(0.1, 1.9),
    Brightness(0.1, 1.9),
    Sharpness(0.1, 1.9),
]


class RandAugmentModule(nn.Module):
    """Trainable `RandAugment` module.

    This class is an implementation of `RandAugment` for tensor inputs. It contains
    trainable magnitude parameter. With differentiable operations, this module can learn
    best performance magnitude value.

    This class performs with randomly selected operations from the given operation set.
    To find the optimal magnitude, you need to apply this augmentations after fetching
    input tensors:

        >>> for inputs, label in train_dataloader:
        >>>     inputs = inputs.cuda()
        >>>     logits = model(augmentor(inputs))
        >>>     ...

    And you should remove other augmentations from `Dataset`:

        >>> transform = Compose([
        >>>     Resize(...),
        >>>     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        >>>     ToTensorV2(),
        >>> ])

    According to some experiments, we observed that the magnitude converges early in
    training. The tensor-version augmentations are much more slower than numpy-version
    ones (because they are performed in parallel, and well-optimized). So after the
    magnitude is converged, you would better to switch the augmentor from
    `RandAugmentModule` to `RandAugment` with the trained magnitude value. You can get
    the magnitude by calling `get_magnitude`.

    In addition, we recommend to use larger learning rate for this module (e.g. 10 times
    larger) in optimization. You can specify different learning rates by following:

        >>> param_groups = [
        >>>    {"params": augmentor.parameters(), "lr": 10 * learning_rate},
        >>>    {"params": model.parameters(), "lr": learning_rate},
        >>> ]
        >>> optimizer = optim.Adam(param_groups)

    Args:
        num_ops: The number of operations.
        normalized: If `True`, the input image is normalized to `[-1, 1]`. In this case,
            this class automatically rescale to `[0, 1]` during the operations and then
            restores to original scale. Default is `True`.
        opset: The operation set to use in `RandAugment`. Default is `DefaultOpSet`.
    """

    def __init__(
        self,
        num_ops: int,
        normalized: bool = True,
        opset: List[BaseOperation] = DefaultOpSet,
    ):
        super().__init__()
        self.num_ops = num_ops
        self.normalized = normalized
        self.opset = opset

        self.magnitude_logits = nn.Parameter(torch.empty(()).normal_(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the image to [0, 1].
        if self.normalized:
            x = (x + 1) / 2

        magnitude = self.magnitude_logits.sigmoid()
        for _ in range(self.num_ops):
            x = random.choice(self.opset)(x, magnitude)

        # Normalize the image to [-1, 1].
        if self.normalized:
            x = 2 * x - 1
        return x

    def get_magnitude(self) -> float:
        """Return the trained magnitude value."""
        return self.magnitude_logits.detach().cpu().sigmoid().item()


class RandAugment(A.ImageOnlyTransform):
    """Integrated image augmentor of `RandAugment`.

    This class is an implementation of `RandAugment` for numpy array inputs. It is based
    on `albumentations` and it can be combined with other operations in the library. For
    example, you can use `RandAugment` by creating:

        >>> transform = Compose([
        >>>     Resize(...),
        >>>     RandAugment(num_ops=..., magnitude=...),
        >>>     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        >>>     ToTensorV2(),
        >>> ])

    And give the above `transform` to your `Dataset`. Then the input images will be
    transformed according to the `RandAugment` policy.

    Contrary to the original paper, the range of magnitude value is `[0, 1]`. It is
    because the magnitude is trained from `RandAugmentModule` and it represents the
    magnitude by applying `sigmoid` to its logit.

    Args:
        num_ops: The number of operations.
        normalized: If `True`, the input image is normalized to `[-1, 1]`. In this case,
            this class automatically rescale to `[0, 1]` during the operations and then
            restores to original scale. Default is `True`.
        opset: The operation set to use in `RandAugment`. Default is `DefaultOpSet`.
    """

    def __init__(
        self, num_ops: int, magnitude: float, opset: List[BaseOperation] = DefaultOpSet
    ):
        super().__init__(always_apply=False, p=1.0)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.opset = opset

    def apply(self, x: np.ndarray, **params: Any) -> np.ndarray:
        for _ in range(self.num_ops):
            op = random.choice(self.opset)
            x = op(x, self.magnitude)
        return x
