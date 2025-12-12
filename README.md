# Medical Information Extraction from Unstructured Text

---

**University of Toronto Department of Electrical and Computer Engineering** 

**Course**: ECE1508 - Applied Deep Learning

---

## Abstract

Textual medical records can be long and time consuming to review, resulting in less time medical professionals can spend offering patient care. The objective of this project was to use a discriminative model to highlight clinically important information using Named Entity Recognition. The model architecture selected was a BiLSTM-CRF, this supervised learning problem required an LLM to initially generate the "true" labels since the initial dataset did not contain any and after further data preprocessing, was used to train the model. After applying regularization and early stop to combat overfitting, the final validation accuracy was 90.95% and the final test accuracy was 90.90%.

---

## Overview

This repository contain a the entire project including the code and project report written in latex. 

### Folder Breakdown

- `Latex Report` contains the project report written in latex
- `Project Code` contain all the code for the project with its own README.md to go its contents specifically
- `.gitignore` specifies the that the virtual enviroment in `Project Code` should be ignored

### Dataset

The dataset used in this project was based on the paper [SimSUM: Simulated Benchmark with Structured and Unstructured Medical Records](https://arxiv.org/abs/2409.08936). The dataset files and implementation resources can be found in the official GitHub repository: https://github.com/prabaey/SimSUM

The dataset is saved in this GitHub repository within the directory: `/Project Code/data/raw/SimSUM.csv`

---

## Disclaimer

`Project code` was written using assistance from LLM generated code