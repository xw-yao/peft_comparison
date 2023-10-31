# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from setuptools import find_packages, setup

extras = {}

setup(
    name="peft_comparison",
    version="0.1.0",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Vijeta Deshpande and Vladislav Lialin",
    author_email="vlad.lialin@gmail.com",
    url="https://github.com/guitaricet/peft_comparison",
    packages=find_packages(),
    python_requires=">=3.10.0",
    install_requires=[
        "numpy>=1.17",
        "torch",
        "tqdm",
        "accelerate",
        "bitsandbytes>=0.41.1",
        "datasets",
        "sentencepiece>=0.1.99",
        "rouge-score>=0.1.2",
        "wandb>=0.15.9",
        "evaluate",
        "scipy",
        "scikit-learn",
        "loguru",
        "nltk",
        "deepspeed",
    ],
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
