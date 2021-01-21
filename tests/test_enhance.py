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

from .utils import assert_enhance_op


def test_AutoContrast_consistency():
    assert_enhance_op(AutoContrast())


def test_Brightness_consistency():
    assert_enhance_op(Brightness(0.1, 1.9))


def test_Color_consistency():
    assert_enhance_op(Color(0.1, 1.9))


def test_Contrast_consistency():
    assert_enhance_op(Contrast(0.1, 1.9))


def test_Equalize_consistency():
    assert_enhance_op(Equalize())


def test_Posterize_consistency():
    assert_enhance_op(Posterize(0, 8, quantize_magnitude=True))


def test_Sharpness_consistency():
    assert_enhance_op(Sharpness(0.1, 1.9))


def test_Solarize_consistency():
    assert_enhance_op(Solarize(0, 0xFF))
