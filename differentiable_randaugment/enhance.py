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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from differentiable_randaugment.base import BaseOperation


class AutoContrast(BaseOperation):
    """Normalize the image contrast.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | AutoContrast  | ✔           | ✘         |
    +---------------+-------------+-----------+

    This class maximize the image contrast by remapping the lowest value to `0` and the
    highest value to `0xFF`. It makes the darkest pixels to be black and the lightest
    ones to be white.

    Note:
        This class does not use the magnitude value. So you do not need to specify the
        range of the magnitude.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        min_val = x.min(axis=(0, 1), keepdims=True)
        max_val = x.max(axis=(0, 1), keepdims=True)

        x = (x - min_val) / (max_val - min_val + 1e-6)
        return np.clip(0xFF * x, 0, 0xFF).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        min_val = x.amin(dim=(2, 3), keepdim=True)
        max_val = x.amax(dim=(2, 3), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-6)


class Invert(BaseOperation):
    """Invert the pixels of the image.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Invert        | ✔           | ✘         |
    +---------------+-------------+-----------+

    This class inverts the image pixels by subtracting them from 0xFF. For the tensors,
    they will be subtracted from `1.0`.

    Note:
        This class does not use the magnitude value. So you do not need to specify the
        range of the magnitude.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return 0xFF - x

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return 1 - x


class Equalize(BaseOperation):
    """Equalize the image histogram.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Equalize      | ✔           | ✘         |
    +---------------+-------------+-----------+

    This class equalize the image histogram. It makes the histogram uniformly by
    applying non-linear mapping to the pixels.

    Since the histogram equalization is not differentiable, we use a trick which is to
    multiply the equalization ratio to the input image. Although the derivative of this
    operation is not strict, it can prevent gradient interruption.

    Note:
        This class does not use the magnitude value. So you do not need to specify the
        range of the magnitude.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return np.stack(
            [cv2.equalizeHist(x[:, :, i]) for i in range(x.shape[-1])], axis=-1
        )

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        origins = (x.detach().cpu().numpy() * 0xFF).astype(np.uint8)

        for i, j in itertools.product(range(x.size(0)), range(x.size(1))):
            ratio = cv2.equalizeHist(origins[i, j]) / (origins[i, j] + 1e-6)
            x[i, j] *= torch.tensor(ratio, dtype=x.dtype, device=x.device)

        return torch.clamp(x, 0, 1)


class Solarize(BaseOperation):
    """Invert all pixels above the threshold.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Solarize      | ✔           | ✘         |
    +---------------+-------------+-----------+

    This class inverts the image pixels which are greater than the magnitude threshold.
    The range of magnitude value is `[0, 0xFF]`. For the tensors, the threshold will be
    divided to be in range `[0, 1]`.

    Note:
        While the magnitude is used as a threshold, the gradient of the magnitude tensor
        cannot be calculated.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return np.where(x < value, x, 0xFF - x)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.where(x < value / 0xFF, x, 1.0 - x)


class SolarizeAdd(BaseOperation):
    """Invert all shifted pixels above the threshold.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | SolarizeAdd   | ✔           | ✔         |
    +---------------+-------------+-----------+

    This class applies brightness to the image and inverts the applied image pixels
    which are greater than the half of the color range (i.e. `0x80` for `byte`s and
    `0.5` for `float`s). The range of magnitude value is `[0, 0xFF]`.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        x = np.clip(x + value, 0, 0xFF).astype(np.uint8)
        return np.where(x < 0x80, x, 0xFF - x)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x + value / 0xFF, 0, 1)
        return torch.where(x < 0.5, x, 1.0 - x)


class Posterize(BaseOperation):
    """Reduce the number of bits for the image pixels.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Posterize     | ✘           | ✔         |
    +---------------+-------------+-----------+

    This class reduces the image quality by quantize pixel values with magnitude level.
    Simply, posterize operation discretizes the continuous values by dropping to certain
    boundary like `floor`. Using this fact, it supports continuous posterize (e.g.
    dropping `4.25` bits).

    In fact, it is hard to consider the gradients for input image and magnitude tensors.
    To make this operation differentiable, we multiply the discretize level (which is
    calculated by the magnitude) to the quantized pixels. Although it is not strict, it
    can somehow affect to the policy training by using the gradient.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return (x & ~(2 ** (8 - int(value)) - 1)).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        shift = 2 ** (8 - value)
        return shift * torch.floor(0xFF * x / shift) / 0xFF


class Contrast(BaseOperation):
    """Control the contrast of the image.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Contrast      | ✔           | ✔         |
    +---------------+-------------+-----------+

    This class adjusts the contrast of images by multiplying the magnitude. It makes
    the image to be black with `magnitude==0` and returns original image with
    `magnitude==1`.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return np.clip(x * value, 0, 0xFF).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * value, 0, 1)


class Color(BaseOperation):
    """Adjust the color balance of the image.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Color         | ✔           | ✔         |
    +---------------+-------------+-----------+

    This class blends the original image and its grayscale image. The magnitude controls
    the blending ratio of them.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return np.clip(
            value * x + (1 - value) * x.mean(2, keepdims=True), 0, 0xFF
        ).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value * x + (1 - value) * x.mean(1, keepdim=True), 0, 1)


class Brightness(BaseOperation):
    """Adjust the brightness of the image.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Brightness    | ✔           | ✔         |
    +---------------+-------------+-----------+

    This class controls the brightness of images by adding the magnitude value to each
    pixel. It makes the image to be black with `magnitude==0` and returns original image
    with `magnitude==1`. That is, the brightness constant is shifted by `1` and
    `magnitude==1` implies `brightness==0`.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        return np.clip(x + (value - 1) * 0xFF, 0, 0xFF).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + value - 1, 0, 1)


class Sharpness(BaseOperation):
    """Adjust the sharpness of the image.

    +---------------+-------------+-----------+
    |               | Input Image | Magnitude |
    +===============+=============+===========+
    | Sharpness     | ✔           | ✔         |
    +---------------+-------------+-----------+

    This class makes the image sharply or blurry. It gives a complete blur image with
    `magnitude==0` and complete sharp one with `magnitude==2`. It performs nothing for
    `magnitude==1` and returns the original image.

    Note:
        This class uses convolution with `5x5` gaussian kernel for applying blur
        operation.
    """

    def apply_numpy(self, x: np.ndarray, value: float) -> np.ndarray:
        blur = cv2.GaussianBlur(x, (5, 5), 3, borderType=cv2.BORDER_CONSTANT)
        return np.clip(value * x + (1 - value) * blur, 0, 0xFF).astype(np.uint8)

    def apply_tensor(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Create a 5x5 gaussian filter.
        kernel = cv2.getGaussianKernel(5, sigma=3)
        kernel = np.outer(kernel, kernel.T).reshape(1, 1, 5, 5)

        kernel = torch.tensor(kernel, dtype=x.dtype, device=x.device)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)

        # Apply the filter and blend with the original image.
        blur = F.conv2d(
            x, kernel, bias=None, stride=1, padding=(2, 2), groups=x.size(1)
        )
        return torch.clamp(value * x + (1 - value) * blur, 0, 1)
