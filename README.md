# <img src="https://github.com/NogaMudrik/SiBBlInGS/blob/main/SIBBLINGS_LOGO2.png?raw=true" width="200" height="200"> 
# SiBBlInGS:
##  Similarity-driven Building-Block Inference using Graphs across States
**NOW AT ICML 2024!!**
This is the code for: Mudrik, N., Mishne, G., & Charles, A. S. (2024). SiBBlinGS: Similarity-driven Building-Block Inference Using Graphs Across States. In Proceedings of the Forty-first International Conference on Machine Learning. [LINK](https://openreview.net/forum?id=h8aTi32tul).


if you use the code, please cite the above paper.


bibtex:

@inproceedings{
mudrik2024sibblings,
title={Si{BB}lIn{GS}: Similarity-driven Building-Block Inference using Graphs across States},
author={Noga Mudrik and Gal Mishne and Adam Shabti Charles},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=h8aTi32tul}
}

### ABSTRACT:
Time series data across scientific domains are often collected under distinct states (e.g., tasks), wherein latent processes (e.g., biological factors) create complex inter- and intra-state variability. A key approach to capture this complexity is to uncover fundamental interpretable units within the data, Building Blocks (BBs), which modulate their activity and adjust their structure across observations. Existing methods for identifying BBs in multi-way data often overlook inter- vs. intra-state variability, produce uninterpretable components, or do not align with properties of real-world data, such as missing samples and sessions of different duration. Here, we present a framework for Similarity-driven Building Block Inference using Graphs across States (SiBBlInGS). SiBBlInGS offers a graph-based dictionary learning approach for discovering sparse BBs along with their temporal traces, based on co-activity patterns and inter- vs. intra-state relationships. Moreover, SiBBlInGS captures per-trial temporal variability and controlled cross-state structural BB adaptations, identifies state-specific vs. state-invariant components, and accommodates variability in the number and duration of observed sessions across states. We demonstrate SiBBlInGS's ability to reveal insights into complex phenomena as well as its robustness to noise and missing samples through several synthetic and real-world examples, including web search and neural data.



### HOW TO USE THE DATA?
The script "run_SiBBlInGS" provides an example of running the code on different examples. The function "run_SiBBlInGS" is the main function to use for training the model.

