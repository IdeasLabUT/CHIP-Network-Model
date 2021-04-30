# Community Hawkes Independent Pairs (CHIP) Network Model

This repo includes the Python implementation of the CHIP network model as well as the code to replicate all experiments 
in the paper [CHIP: A Hawkes Process Model for Continuous-time Networks with Scalable and Consistent Estimation](https://arxiv.org/abs/1908.06940), 
presented at **Neural Information Processing Systems (NeurIPS) 2020**.

## Introduction
CHIP is a generative model for continuous-time networks of *timestamped relational events*, where each 
event is a triplet (i, j, t) denoting events from node i (sender) to node j (receiver) at timestamp t.

Main contributions:
1. We demonstrate that spectral clustering provides **consistent community detection** in CHIP for a growing number of nodes.
2. We propose **consistent and computationally efficient estimators** for the model parameters for a growing number of nodes 
and time duration.
3. We show that CHIP provides better fits to several real datasets and **scales to much larger networks** than 
existing models, including a Facebook network with over 40,000 nodes and over 800,000 events.


## Setup
This repo has been developed and tested using Python 3.6.9. The code **does not** work with Python 2.7.

To run experiments, either clone or fork this repository and refer to [requirements.txt](https://github.com/IdeasLabUT/CHIP-Network-Model/blob/master/requirements.txt) 
for the required packages.

## Datasets
All datasets used in this repo are either available in the [storage/datasets](https://github.com/IdeasLabUT/CHIP-Network-Model/tree/master/storage/datasets)
directory or will be automatically downloaded by the preprocessing script.


## Examples
There are 3 Jupyter notebook examples of the CHIP model in the [examples](https://github.com/IdeasLabUT/CHIP-Network-Model/tree/master/examples) directory:

- [generating_chip_networks.ipynb](https://github.com/IdeasLabUT/CHIP-Network-Model/blob/master/examples/generating_chip_networks.ipynb): how to generate and fit networks using CHIP
- [facebook_wallposts_exploratory_analysis.ipynb](https://github.com/IdeasLabUT/CHIP-Network-Model/blob/master/examples/facebook_wallposts_exploratory_analysis.ipynb): fit CHIP to the largest connected component of the Facebook Wall Post dataset
- [enron_exploratory_analysis.ipynb](https://github.com/IdeasLabUT/CHIP-Network-Model/blob/master/examples/enron_exploratory_analysis.ipynb): fit CHIP to the Enron dataset


## Contact
Please contact us if you have any questions or to report an issue. You can find the contact information of all three 
authors in the [paper](https://arxiv.org/abs/1908.06940).

> This repository has been published for the sole purpose of providing more information on the aforementioned publication.