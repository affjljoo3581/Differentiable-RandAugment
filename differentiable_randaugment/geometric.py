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

import itertools
import random
from abc import ABCMeta, abstractmethod
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from differentiable_randaugment.base import BaseOperation

TransformMatrix = List[List[Union[float, torch.Tensor]]]


class GeometricOperation(BaseOperation, metaclass=ABCMeta):
    """Template class for geometric transform operations.

    Usually the geometric operations use affine transformation. This class performs the
    image augmentation by applying certain affine transform matrix. All geometric
    operations which inherit this class must implement `transform` method which returns
    the affine matrix. The input `value` must be differentiable if it is torch tensor.

    Note:
        The background color of transformed images is `#808080` (or `0.5` for tensors
        which are in range `[0, 1]`).
    """

    @abstractmethod
    def transform(self, value: Union[float, torch.Tensor]) -> TransformMatrix:
        ...

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        height, width = x.shape[:2]

        # Since `cv2` apply affine transformation by fixing center point to the left-top
        # corner, the coordinate normalization is required.
        normalize = np.array([[2 / width, 0, -1], [0, 2 / height, -1], [0, 0, 1]])
        matrix = self.transform(value) + [[0, 0, 1]]
        matrix = np.linalg.inv(matrix @ normalize) @ normalize

        return cv2.warpAffine(
            x,
            matrix[:2, :],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=(0x80, 0x80, 0x80),
        )

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros((2, 3), dtype=x.dtype, device=x.device)

        # To make the matrix differentiable, we copy all values to the empty matrix
        # tensor directly.
        transform = self.transform(value)
        for i, j in itertools.product(range(2), range(3)):
            matrix[i, j] = transform[i][j]

        matrix = matrix.expand(x.size(0), -1, -1)
        grid = F.affine_grid(matrix, x.shape, align_corners=True)

        # Unfortunately, torch does not support background color in affine
        # transformation. Instead, it fills the background with zero. So we will
        # normalize the input tensor to make the background color be `0.5`.
        x = F.grid_sample(2 * x - 1, grid, "bilinear", "zeros", align_corners=True)
        return (x + 1) / 2


class ShearX(GeometricOperation):
    """Apply shear operation along the horizontal axis.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | ShearX        | ✔           | ✔         |
    +---------------+-------------+-----------+

    The magnitude of this operation must be positive and relative. It performs shear
    operation with entire width pixels for `magnitude==1` and half pixels for
    `magnitude==0.5`. Note that the shear direction is randomly selected.
    """

    def transform(self, sx: Union[float, torch.Tensor]) -> TransformMatrix:
        sx = random.choice([-1, 1]) * sx * 2
        return [[1, sx, 0], [0, 1, 0]]


class ShearY(GeometricOperation):
    """Apply shear operation along the vertical axis.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | ShearY        | ✔           | ✔         |
    +---------------+-------------+-----------+

    The magnitude of this operation must be positive and relative. It performs shear
    operation with entire height pixels for `magnitude==1` and half pixels for
    `magnitude==0.5`. Note that the shear direction is randomly selected.
    """

    def transform(self, sy: Union[float, torch.Tensor]) -> TransformMatrix:
        sy = random.choice([-1, 1]) * sy * 2
        return [[1, 0, 0], [sy, 1, 0]]


class TranslateX(GeometricOperation):
    """Translate the image in the horizontal direction.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | TranslateX    | ✔           | ✔         |
    +---------------+-------------+-----------+

    A magnitude of this operation must be positive and relative. It performs translate
    operation with entire width pixels for `magnitude==1` and half pixels for
    `magnitude==0.5`. Note that the translate direction is randomly selected.
    """

    def transform(self, tx: Union[float, torch.Tensor]) -> TransformMatrix:
        tx = random.choice([-1, 1]) * tx * 2
        return [[1, 0, tx], [0, 1, 0]]


class TranslateY(GeometricOperation):
    """Translate an image in the vertical direction.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | TranslateX    | ✔           | ✔         |
    +---------------+-------------+-----------+

    A magnitude of this operation must be positive and relative. It performs translate
    operation with entire height pixels for `magnitude==1` and half pixels for
    `magnitude==0.5`. Note that the translate direction is randomly selected.
    """

    def transform(self, ty: Union[float, torch.Tensor]) -> TransformMatrix:
        ty = random.choice([-1, 1]) * ty * 2
        return [[1, 0, 0], [0, 1, ty]]


class Rotate(GeometricOperation):
    """Rotate an image clockwise and counter-clockwise.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | TranslateX    | ✔           | ✔         |
    +---------------+-------------+-----------+

    A magnitude of this operation must be positive and in degrees. The rotation
    direction is randomly selected (clockwise or counter-clockwise).
    """

    def transform(self, deg: Union[float, torch.Tensor]) -> TransformMatrix:
        if not isinstance(deg, torch.Tensor):
            deg = torch.tensor(deg, dtype=torch.float)

        rad = random.choice([-1, 1]) * torch.deg2rad(deg)
        alpha, beta = torch.cos(rad), torch.sin(rad)

        return [[alpha, -beta, 0], [beta, alpha, 0]]
