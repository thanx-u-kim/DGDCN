# DGDCN (Reproduction Code)

This repository contains the code used in our paper:

> "Incorporating Traffic Flow Propagation Impacts into Dynamic Graph Diffusion Convolutional Network for Traffic Speed Prediction", H. Shon., S. Kim., Y. Shin., Y. Yoon., J. Lee., 2025  

## ✳️ Data Availability
Due to security and privacy restrictions, the dataset used in the paper **cannot be shared**.  
Users may reproduce results with their own dataset following the same format:
- `feat`: CSV file, shape (T, N)
- `in_adj`, `out_adj`: N×N numpy adjacency matrices (`.npy`)
- `dijkstra`: N×N CSV distance matrix

> **Note**: Paths in `main.ipynb` assume `../input_data/` by default.  
> Replace with your local paths.

## Requirements
```bash
pip install -r requirements.txt
