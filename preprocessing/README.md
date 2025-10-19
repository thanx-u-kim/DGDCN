## Preprocessing Pipeline

This folder contains five notebooks that convert raw taxi DTG logs into model-ready inputs for experiments.  
The canonical order is:

**1 → 2 → (3, 4, 5 in any order)**

After Step 2 finishes (map-matching & graph alignment), Steps 3–5 can run independently.

---

### Step 1 — DTG Cleaning (`01_DTG_Cleaning.ipynb`)
**Goal:** Load raw DTG logs and standardize trajectories.

**Key operations:**  
Clean raw DTG data by removing outliers, idling, and invalid GPS points.

**Inputs:**  
Raw DTG files.  
> ⚠️ *Note:* The format and structure of DTG data may vary depending on individuals.
> Users should adjust column names, delimiters, and preprocessing logic according to their own DTG file format.

**Outputs:**  
Standardized per-trip data and a trajectory dictionary (saved per day).

---

### Step 2 — Map-Matching & Graph Networks (`02_Mapmatching_and_Graph_Networks.ipynb`)
**Goal:** Map-match trajectories to the road network and derive graph-level features.

**Key operations:**  
Match DTG trajectories to network edges and compute per-link speed, connectivity, and inflow/outflow adjacency matrices.

**Inputs:**  
Outputs from Step 1

**Outputs:**  
Map-matched samples and per-slice artifacts (`in_flow`, `out_flow`, `speed` pickles).

---

### Step 3 — Dynamic Flow Adjacency Matrices (`03_Compute_Flow_Matrices.ipynb`)
**Goal:** Aggregate per-slice flow pickles into time-varying adjacency tensors.

**Key operations:**  
Combine inflow and outflow data across time slices to generate `(T, N, N)` adjacency matrices.

**Inputs:**  
Step 2 outputs (`./Mapmatching_output/...`).

**Outputs:**  
`inflow_adj_data.npy`, `outflow_adj_data.npy`.

---

### Step 4 — Dijkstra Distance Matrix (`04_Compute_Dijkstra_Distances.ipynb`)
**Goal:** Build an node-to-node shortest-path distance matrix for the road graph.

**Key operations:**  
Compute all-pairs shortest-path distances using Dijkstra’s algorithm.

**Inputs:**  
Any single `speed_*.pkl` file generated from Step 2 can be used, as it contains the necessary network structure and edge length information for constructing the distance graph.

**Outputs:**  
`dijkstra_matrix.csv` (`N × N` distance matrix).

---

### Step 5 — Speed Data Assembly (`05_Assemble_Speed_Data.ipynb`)
**Goal:** Aggregate per-slice link speeds into a `(T, N)` feature matrix.

**Key operations:**  
Merge per-slice speed data, fill missing values using time-based interpolation, and export the unified tensor.

**Inputs:**  
Step 2 outputs (`./Mapmatching_output/...`).

**Outputs:**  
`speed_data.csv` (`T × N`).

---

### Final Artifacts (consumed by the model)
- `input_data/speed_data.csv` — `(T, N)`
- `input_data/inflow_adj_data.npy` — `(T, N, N)`
- `input_data/outflow_adj_data.npy` — `(T, N, N)`
- `input_data/dijkstra_matrix.csv` — `(N, N)`
