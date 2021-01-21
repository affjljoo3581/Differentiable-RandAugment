from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

builtins.__PYTHON_SETUP__ = True
import differentiable_randaugment as dra  # noqa isort:skip


setup(
    name="differentiable_randaugment",
    version=dra.__version__,
    author=dra.__author__,
    author_email=dra.__author_email__,
    description=dra.__doc__,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=[
        "dataset",
        "deep-learning",
        "machine-learning",
        "augmentation",
        "image",
        "computer-vision",
        "overfitting",
    ],
    url=dra.__homepage__,
    license=dra.__license__,
    packages=find_packages(exclude=["tests", "tests/*"]),
    python_requires=">=3.7",
    install_requires=["opencv_python", "torch>=1.7", "albumentations", "numpy"],
    extras_require={"tests": ["pytest"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
