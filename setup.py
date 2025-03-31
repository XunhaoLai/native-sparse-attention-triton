# Copyright 2025 Xunhao Lai & Jianqiao Lu.
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

from setuptools import setup, find_packages

# Read the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the setup configuration
setup(
    name="native-sparse-attention-triton",
    version="0.1.0",
    description="An efficient implementation of Native Sparse Attention using Triton",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="XunhaoLai",
    author_email="laixunhao@pku.edu.cn",  # Replace with your actual email
    url="https://github.com/XunhaoLai/native-sparse-attention-triton",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "triton>=3.0.0",
        "einops>=0.7.0",
        "flash-attn>=2.6.3",
        "transformers>=4.44.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
