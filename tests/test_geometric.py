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

from differentiable_randaugment.geometric import (
    Cutout,
    Rotate,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
)

from .utils import assert_geometric_op


def test_Rotate_consistency():
    assert_geometric_op(Rotate(0, 45))


def test_ShearX_consistency():
    assert_geometric_op(ShearX(0, 0.5))


def test_ShearY_consistency():
    assert_geometric_op(ShearY(0, 0.5))


def test_TranslateX_consistency():
    assert_geometric_op(TranslateX(0, 0.5))


def test_TranslateY_consistency():
    assert_geometric_op(TranslateY(0, 0.5))


def test_Cutout_consistency():
    assert_geometric_op(Cutout(0, 0.2))
