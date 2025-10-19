# Dynamic Graph Diffusion Convolutional Network for Traffic Speed Prediction(DGDCN)

Implementation of **DGDCN (Dynamic Graph Diffusion Convolutional Network)** in PyTorch.  
This repository provides code and example setup for reproducing our experiments.

## 📌 Publication Status
> ⚠️ **Note:**  
> This work is based on our manuscript currently under **major revision** for submission to a peer-reviewed journal.  
> The paper is **not yet published**, and details may change in the final accepted version.  
> The README and repository will be updated once the manuscript is accepted.

## Requirements
Our code is based on Python 3.8.13).
Tested on **PyTorch 1.11.0 (torch==1.11.0, CUDA 11.3)**.
Major dependencies are listed below:
- python==3.8.13
- numpy==1.22.4
- pandas==1.4.2
- torch==1.11.0 # PyTorch 1.11.0
- pytorch-lightning==1.6.4
- torchmetrics==0.9.0
```bash
pip install -r requirements.txt
```

## ✳️ Data Availability
Due to security and privacy restrictions, the dataset used in the paper **cannot be shared**.
Users may reproduce results with their own dataset following the same format.

### 📂 Required Files and Shapes
- `speed_data.csv` (`feat`):
  - Shape: **(T x N)**
- `inflow_adj_data.npy` & `outflow_adj_data.npy` (`in_adj`, `out_adj`):
  - Shape: **(T x N x N)**
- `dijkstra_matrix.csv` (`dijkstra`):
  - Shape: **(N x N)**
  - Shortest path distance matrix based on the road network topology

### 📌 Notes
- **T(Total Timesteps)** = total number of time steps.  
  - Example in our setting: `2880` (20 days × 144 steps per day, with 10-minute resolution).  
- **N(Number of Nodes)** = the number of road sections in the study area.  
- Data should be placed in `../input_data/` by default, or paths can be adjusted in the code.

### 📑 Example Code
```python
# Data Loading
feat = utils.load_dataset('../input_data/speed_data.csv')
in_adj = utils.load_adjacency('../input_data/inflow_adj_data.npy')
out_adj = utils.load_adjacency('../input_data/outflow_adj_data.npy')
dijkstra = utils.load_dijkstra('../input_data/dijkstra_matrix.csv')
```

## 🧩 Preprocessing Pipeline

If you have your own DTG data, use the pipeline in [`preprocessing/`](./preprocessing) first. It converts raw taxi DTG logs into the four artifacts used by the main notebook.

### Overview
The canonical order is: **1 → 2 → (3, 4, 5 in any order)**.  
After Step 2 (map-matching & graph networks) is done, Steps 3–5 are independent and can run in parallel.

> **DTG format note.** DTG files differ across providers. Adjust column names, delimiters, and cleaning rules to your schema.

---

### Steps & I/O at a glance

**Step 1 — DTG Cleaning** (`01_DTG_Cleaning.ipynb`)  
**Goal:** Standardize trajectories from raw DTG.  
**Does:** Remove outliers/idling/invalid GPS; produce per-trip/day structures.  
**In:** Raw DTG files.  
**Out:** Cleaned per-trip data & a trajectory dictionary.

**Step 2 — Map-Matching & Graph Networks** (`02_Mapmatching_and_Graph_Networks.ipynb`)  
**Goal:** Map-match trajectories to road links and derive link-level metrics.  
**Does:** Match to road links; compute per-link speed, connectivity; emit per-slice `in_flow`, `out_flow`, `speed` pickles.  
**In:** Step 1 outputs.  
**Out:** Per-slice speed, inflow & outflow adjacency artifacts (pickles).

**Step 3 — Dynamic Flow Adjacency Matrices** (`03_Compute_Flow_Matrices.ipynb`)  
**Goal:** Build time-varying inflow/outflow adjacency tensors.  
**Does:** Aggregate per-slice flow to `(T, N, N)`.  
**In:** Step 2 outputs (`./Mapmatching_output/...`).  
**Out:** `inflow_adj_data.npy`, `outflow_adj_data.npy` (shape `T × N × N`).

**Step 4 — Dijkstra Distance Matrix** (`04_Compute_Dijkstra_Distances.ipynb`)  
**Goal:** All-pairs shortest-path distance on the road graph.  
**Does:** Compute Dijkstra distances for every node pair.  
**In:** **Any single** `speed_*.pkl` from Step 2 (contains the network/length info needed).  
**Out:** `dijkstra_matrix.csv` (shape `N × N`).

**Step 5 — Speed Data Assembly** (`05_Assemble_Speed_Data.ipynb`)  
**Goal:** Create the unified `(T, N)` speed feature matrix.  
**Does:** Merge per-slice speeds, fill gaps (time-based interpolation), export CSV.  
**In:** Step 2 outputs (`./Mapmatching_output/...`).  
**Out:** `speed_data.csv` (shape `T × N`).

---

### Final artifacts (consumed by the main notebook)
Place these under `input_data/` (or adjust the paths in the notebook):

## Run (Notebook)
1. Open the `main.ipynb` file.  
2. Execute all cells in order from top to bottom.  
3. Make sure the data path (`../input_data/`) matches your dataset location and adjust if necessary.  
4. Check the training arguments (`args`). The notebook provides default values for our model, but you may adjust them as needed.  
5. Running all cells will start training, save checkpoints, and produce outputs.  
6. At the end of the notebook, evaluation results (e.g., RMSE, MAE, MAPE) and sample predictions will be displayed.
