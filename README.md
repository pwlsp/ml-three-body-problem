# Three-Body Problem ML Challenge

## 📌 Description
This project is part of a course assignment.  
The goal is to build a machine learning model that can predict the trajectories of three bodies in a gravitational system **without using numerical solvers**.  
The task is based on simulated datasets of the Three-Body Problem, provided in a Kaggle competition.

## What has been done so far
### Task 1.1 Data Preparation and Validation Pipeline

- **Trajectory structure:** The raw dataset is organized in fixed-length trajectories of **258 rows** each (one row per time step).
- **Zero-tail detection & cleaning:** After a collision event, all state coordinates in a trajectory become zeros from the first all-zero row up to the end (row 258).  
  We detect the first all-zero row per trajectory and **drop that row and everything after it**, producing a cleaned dataframe `df_clean` without zero tails.
- **Visualization:** We plot sample trajectories from `df_clean` (i.e., after removing zero tails) to sanity-check geometry and continuity.
- **Validation split (no leakage):**  
  - We **shuffle unique `traj_id` values** and split them into **train (70%)**, **validation (15%)**, and **test (15%)**.  
  - Each trajectory is kept **entirely within a single split** to avoid leakage across sets.
- **Tiny subset for fast iteration:**  
  - We build a **tiny dataset (~2% of all rows)** by sampling **whole trajectories** within each split until we reach the target row count per split (proportional to the full distribution).  
  - This preserves train/val/test proportions and still guarantees that no trajectory is split across sets.

> Note: The 258-row length applies to the **raw** trajectories. After removing zero tails, trajectories in `df_clean` can be shorter than 258 rows, by design.

