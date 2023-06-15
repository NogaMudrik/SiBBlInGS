# SiBBlInGS
This is the code for: Mudrik, Noga, Gal Mishne, and Adam S. Charles. "SiBBlInGS: Similarity-driven Building-Block Inference using Graphs across States." arXiv preprint arXiv:2306.04817 (2023). [LINK](https://arxiv.org/abs/2306.04817).

if you use the code, please cite the above paper.

ABSTRACT:
Interpretable methods for extracting meaningful building blocks (BBs) underlying multi-dimensional time series are vital for discovering valuable insights in complex systems. Existing techniques, however, encounter limitations that restrict their applicability to real-world systems, like reliance on orthogonality assumptions, inadequate incorporation of inter- and intra-state variability, and incapability to handle sessions of varying duration. Here, we present a framework for Similarity-driven Building Block Inference using Graphs across States (SiBBlInGS). SiBBlInGS employs a graph-based dictionary learning approach for BB discovery, simultaneously considers both inter- and intra-state relationships in the data, can extract non-orthogonal components, and allows for variations in session counts and duration across states. Additionally, SiBBlInGS allows for cross-state variations in BB structure and per-trial temporal variability, can identify state-specific vs state-invariant BBs, and offers both supervised and data-driven approaches for controlling the level of BB similarity between states. We demonstrate SiBBlInGS on synthetic and real-world data to highlight its ability to provide insights into the underlying mechanisms of complex phenomena and its applicability to data in various fields.



HOW TO USE THE DATA?
The script "run_SiBBlInGS" provides an example of running the code on different examples. The function "run_SiBBlInGS" is the main function to use for training the model.

