import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="z-quantum-qaoa",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="QAOA implementation for Orquestra.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/z-quantum-qaoa ",
    packages=["zquantum.qaoa"],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "z-quantum-core",
        "forestopenfermion @ http://github.com/zapatacomputing/forest-openfermion/tarball/fast-op-conversion",
    ],
)
