# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

install_requires = [
    "cugraph",
    "numba>=0.56.2",
    "numpy",
]

setup(
    name="cugraph-dgl",
    description="cugraph wrappers around DGL",
    version="23.04.00",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
    ],
    author="NVIDIA Corporation",
    url="https://github.com/rapidsai/cugraph",
    packages=find_packages(include=["cugraph_dgl*"]),
    install_requires=install_requires,
    license="Apache",
    zip_safe=True,
)
