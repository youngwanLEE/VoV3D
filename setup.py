#!/usr/bin/env python3
# Copyright Youngwan Lee (ETRI). All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="vov3d",
    version="1.0",
    author="Youngwan Lee",
    url="unknown",
    description="VoV3D for efficient video understanding",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "detectron2",
        "opencv-python",
        "pandas",
        "torchvision>=0.4.2",
        "sklearn",
        "tensorboard",
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
