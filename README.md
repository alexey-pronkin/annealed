# annealed

Implementation and reproducing paper "Improving Explorability in Variational Inference with Annealed Variational Objectives Bayesian methods" https://arxiv.org/abs/1809.01818

## Team

- Aleksei Pronkin
- Mikhail Kurenkov
- Timur Chikichev

## Goal

During this project we are going to implement methods and repeat experiments of this paper. 

## Experiments

- Biased noise model.
- Toy energy fitting.
- Quantitative analysis on robustness to beta annealing.
- Amortized inference on MNIST and CelebA datasets.

## Models

- VAE https://arxiv.org/abs/1312.6114 with parameters from https://github.com/jmtomczak/vae_householder_flow
- HVI based on https://arxiv.org/abs/1511.02386.
- IWAE based on https://arxiv.org/abs/1509.00519.

## Pipline

- Implement AVO method https://arxiv.org/abs/1809.01818.
- Implement models.
- Analyze a behavior of AVO objective on toy examples.
- Conduct experiments with VAE and AVO on the MNIST and CelebA datasets.

We planned to use and rewrite some code from https://github.com/joelouismarino/iterative_inference/, https://github.com/jmtomczak/vae_householder_flow, https://github.com/AntixK/PyTorch-VAE, https://github.com/haofuml/cyclical_annealing and https://github.com/ajayjain/lmconv. (We assume, that first two repositories were used in the original paper closed source code) 

