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

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseOperation(ABC):
    """Abstract class of image augmentations.

    This class is a base interface for image augmentations. All image augmentors must
    inherit this class and implement `apply_numpy` and `apply_tensor` methods.

    According to the original paper of `RandAugment`, all magnitudes of operations are
    controlled by a master magnitude which has its own range. The operations use the
    master magnitude by rescaling to their own ranges. This class manages it by
    rescaling the input master magnitude to the given range automatically.

    Args:
        min_val: The minimum magnitude for each operation.
        max_val: The maximum magnitude for each operation.
        quantize_magnitude: If `True`, quantize the magnitude by converting the scaled
            value to `int` type. Default is `False`.

    Note:
        If you do not specify the magnitude range, the master magnitude will be passed
        directly without scaling.
    """

    def __init__(self, *args: float, quantize_magnitude: bool = False):
        if len(args) != 0 and len(args) != 2:
            raise ValueError(
                f"expected 0 or 2 arguments, but given {len(args)} arguments"
            )

        self.magnitude_range = args
        self.quantize_magnitude = quantize_magnitude

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor], magnitude: Union[float, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Perform an image augmentation.

        This function detects a type of an input image and apply a corresponding
        transformation. For instance, if you feed a numpy array image, `apply_numpy`
        method will be performed. And if a torch tensor image is passed, `apply_tensor`
        method will be performed contrastively.

        Basically, all operations support differentiation of the input image and
        magnitude tensor. So you can calculate gradients of the image and magnitude
        tensors respectively.

        Args:
            x: The input image. It should be numpy array or torch tensor.
            magnitude: The master magnitude. It must be in range [0, 1]. If the input
                image is a tensor, it must also be a tensor.

        Returns:
            The augmented image. It will have same type of the input.
        """
        if self.magnitude_range:
            min_val, max_val = self.magnitude_range
            magnitude = magnitude * (max_val - min_val) + min_val

        if self.quantize_magnitude:
            if isinstance(magnitude, torch.Tensor):
                magnitude = magnitude.long()
            else:
                magnitude = int(magnitude)

        if isinstance(x, np.ndarray):
            return self.apply_numpy(x, magnitude)
        elif isinstance(x, torch.Tensor):
            return self.apply_tensor(x, magnitude)
        else:
            raise TypeError(f"type {type(x)} is not allowed")

    @abstractmethod
    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        ...

    @abstractmethod
    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        ...


class Identity(BaseOperation):
    """This operation does nothing. It returns the input image intactly."""

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return x

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return x
