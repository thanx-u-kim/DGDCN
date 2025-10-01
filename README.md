# Dynamic Graph Diffusion Convolutional Network for Traffic Speed Prediction(DGDCN)

Implementation of **DGDCN (Dynamic Graph Diffusion Convolutional Network)** in PyTorch.  
This repository provides code and example setup for reproducing our experiments.

---
## 📌 Publication Status
> ⚠️ **Note:**  
> This work is based on our manuscript currently under **major revision** for submission to a peer-reviewed journal.  
> The paper is **not yet published**, and details may change in the final accepted version.  
> The README and repository will be updated once the manuscript is accepted.
---

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
