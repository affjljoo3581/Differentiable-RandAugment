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

__version__ = "0.1.1"
__author__ = "affjljoo3581"
__author_email__ = "affjljoo3581@gmail.com"
__doc__ = "Optimize RandAugment with differentiable operations"
__homepage__ = "https://github.com/affjljoo3581/Differentiable-RandAugment"
__license__ = "Apache-2.0"


try:
    _ = None if __PYTHON_SETUP__ else None
except NameError:
    __PYTHON_SETUP__ = False


if __PYTHON_SETUP__:
    pass
else:
    from differentiable_randaugment.randaugment import RandAugment, RandAugmentModule
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

    __all__ = [
        "RandAugment",
        "RandAugmentModule",
        "AutoContrast",
        "Brightness",
        "Color",
        "Contrast",
        "Equalize",
        "Posterize",
        "Sharpness",
        "Solarize",
        "Rotate",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
