from setuptools import find_packages, setup

__version__ = "0.0.2"
URL = "https://github.com/predict-idlab/RR-GCN"

install_requires = ["tqdm", "numpy", "scikit-learn"]
setup(
    name="rrgcn",
    version=__version__,
    description="Random Relational Graph Convolutional Networks",
    author="Vic Degraeve",
    author_email="vic.degraeve@ugent.be",
    url=URL,
    keywords=[
        "representation-learning",
        "pytorch",
        "knowledge-graph-embedding",
        "random-relational-graph-convolutional-networks",
        "relational-graph-convolutional-networks",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
