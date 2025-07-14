#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="diffusion_frameworks",
    version="0.0.1",
    description="Implementation of the EDM framework on the ImageNet-256 dataset.",
    author="Grégoire Mourre",
    author_email="gregoire.mourre@gmail.com",
    url="",
    install_requires=["lightning", "hydra-core"],
    package_dir={"": "src"},
    packages=find_packages("src"),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
