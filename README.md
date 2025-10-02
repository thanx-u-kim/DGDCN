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
Users may reproduce results with their own dataset following the same format.

### 📂 Required Files and Shapes
- `speed_data.csv` (`feat`):
  - Shape: **(Total Timesteps x Number of Nodes)**
- `inflow_adj_data.npy` & `outflow_adj_data.npy` (`in_adj`, `out_adj`):
  - Shape: **(Total Timesteps x Number of Nodes x Number of Nodes)**
- `dijkstra_matrix.csv` (`dijkstra`):
  - Shape: **(Number of Nodes x Number of Nodes)**
  - Shortest path distance matrix based on the road network topology

### 📌 Notes
- **Total Timesteps** = total number of time steps.  
  - Example in our setting: `2880` (20 days × 144 steps per day, with 10-minute resolution).  
- **Number of Nodes** = number of locations (road network nodes) in the study area.  
- Data should be placed in `../input_data/` by default, or paths can be adjusted in the code.

### 📑 Example Code
```python
# Data Loading
feat = utils.load_dataset('../input_data/speed_data.csv')
in_adj = utils.load_adjacency('../input_data/inflow_adj_data.npy')
out_adj = utils.load_adjacency('../input_data/outflow_adj_data.npy')
dijkstra = utils.load_dijkstra('../input_data/dijkstra_matrix.csv')
```

## Requirements
```bash
pip install -r requirements.txt
