# Differentiable RandAugment
**Optimize RandAugment with differentiable operations**

![build](https://github.com/affjljoo3581/Differentiable-RandAugment/workflows/build/badge.svg)
![PyPI](https://img.shields.io/pypi/v/differentiable_randaugment)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/differentiable_randaugment)
![PyPI - Format](https://img.shields.io/pypi/format/differentiable_randaugment)
![PyPI - License](https://img.shields.io/pypi/l/differentiable_randaugment?color=blue)
[![codecov](https://codecov.io/gh/affjljoo3581/Differentiable-RandAugment/branch/master/graph/badge.svg?token=3VSK8ZF367)](https://codecov.io/gh/affjljoo3581/Differentiable-RandAugment)
[![CodeFactor](https://www.codefactor.io/repository/github/affjljoo3581/differentiable-randaugment/badge)](https://www.codefactor.io/repository/github/affjljoo3581/differentiable-randaugment)

## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Getting Started](#getting-started)
  - [Support Operations](#support-operations)
  - [License](#license)

## Introduction
**Differentiable RandAugment** is a differentiable version of [RandAugment](https://arxiv.org/abs/1909.13719). The original paper proposed to find optimal parameters by using [**grid search**](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search). Instead, this library supports differentiable operations to calculate gradient of the magnitude parameter and optimize it. See [getting started](#getting-started).

## Installation

To install the latest version from PyPI:

    $ pip install -U differentiable_randaugment

Or you can install from source by cloning the repository and running:

    $ git clone https://github.com/affjljoo3581/Differentiable-RandAugment.git
    $ cd Differentiable-RandAugment
    $ python setup.py install

## Dependencies
- opencv_python
- torch>=1.7
- albumentations
- numpy

## Getting Started

First, create `RandAugmentModule` with your desired number of operations. This module is a differentiable and `torch.Tensor` calculable version of `RandAugment` policy. Using this module, you can train the policy as one of the neural-network model. Note that randomly selected `num_ops` operations will be applied to the images.

```python
  from differentiable_randaugment import RandAugmentModule

  augmentor = RandAugmentModule(num_ops=2)
```

Now you need to perform the module to the images. Usually augmentations are applied in `Dataset`. That is, the operations use `np.ndarray` images. However, it cannot calculate the gradients for image and magnitude parameter (because the entire optimization procedure is based on `torch.Tensor`s). To resolve this, you should apply this module to `torch.Tensor` images rather than `np.ndarray`.

```python
  for inputs, labels in train_dataloader:
      inputs = inputs.cuda()
      logits = model(augmentor(inputs))
      ...
```

Of course, other augmentations should be removed from preprocessing:

```python
  transform = Compose([
      Resize(...),
      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ToTensorV2(),
  ])
```

And lastly, create an optimizer with this module parameters. We recommend to use different learning rate for the model and the augmentor:

```python
  param_groups = [
      {"params": augmentor.parameters(), "lr": 10 * learning_rate},
      {"params": model.parameters(), "lr": learning_rate},
  ]
  optimizer = optim.Adam(param_groups)
```

Now the `RandAugment` policy will be trained with your prediction model.

After training `RandAugmentModule`, get the trained optimal magnitude value by calling `augmentor.get_magnitude()` and use the magnitude as follows:

```python
  from differentiable_randaugment import RandAugment

  transform = Compose([
      Resize(...),
      RandAugment(num_ops=..., magnitude=...),
      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ToTensorV2(),
  ])
  dataset = Dataset(..., transform=transform)
```

While `RandAugment` is an extension of `albumentations`, you can combine other augmentations in `albumentations` with this class.

## Support Operations

**Differentiable RandAugment** supports 14 operations described in the original paper. The below table shows the detailed differential specification of each operation.

|               | Input Image | Magnitude |
|---------------|:-----------:|:---------:|
| Identity      | ✔ |   |
| ShearX        | ✔ | ✔ |
| ShearY        | ✔ | ✔ |
| TranslateX    | ✔ | ✔ |
| TranslateY    | ✔ | ✔ |
| Rotate        | ✔ | ✔ |
| AutoContrast  | ✔ |    |
| Equalize      | ✔ |    |
| Solarize      | ✔ |    |
| Posterize     |    | ✔ |
| Contrast      | ✔ | ✔ |
| Color         | ✔ | ✔ |
| Brightness    | ✔ | ✔ |
| Sharpness     | ✔ | ✔ |

## License
**Differentiable RandAugment** is [Apache-2.0 Licensed](/LICENSE).
