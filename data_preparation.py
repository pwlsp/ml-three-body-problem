# Loading data from data/X_train.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from math import comb

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone


DATA_DIR = Path("data") 
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = DATA_DIR / "X_train.csv"
TEST_PATH = DATA_DIR / "X_test.csv"


def index_dfs_trajectories(_df):
    # Calculate the number of trajectories based on actual data
    n_traj = len(_df) // 257
    remainder = len(_df) % 257

    if remainder == 0:
        _df["traj_id"] = np.repeat(np.arange(n_traj), 257)
    else: 
        # Handle incomplete last trajectory
        complete_ids = np.repeat(np.arange(n_traj), 257)
        incomplete_ids = np.full(remainder, n_traj)
        _df["traj_id"] = np.concatenate([complete_ids, incomplete_ids])


def mark_zero_tails(_df):
    # columns used for zero-tail check (we take only coordinates columns)
    state_cols = [c for c in _df.columns if c not in ["t", "Id", "traj_id"]]

    # mark zero-tail per trajectory (first all-zero row and everything after)
    # we add a column with true or false
    # true means that the row is after collision and we can delete it
    def mark(g):
        z = (g[state_cols] == 0).all(axis=1).to_numpy()
        if not z.any():
            g["zero_tail"] = False
            return g
        first_zero = np.argmax(z)
        mask = np.zeros(len(g), dtype=bool)
        mask[first_zero:] = True
        g["zero_tail"] = mask
        return g

    _df = _df.groupby("traj_id", group_keys=False).apply(mark)

    return _df


def load_process_and_clean_csv(_path):
    df = pd.read_csv(_path)
    index_dfs_trajectories(df)
    df = mark_zero_tails(df) 
    df_clean = df[~df["zero_tail"]].copy() # Drop post-collision tails

    return df_clean


def generate_df_subset(df, frac, rng):
    n_total = len(df)
    target_rows = max(1, int(n_total * frac))
    traj_lengths = df.groupby("traj_id").size().to_dict()
    
    # Shuffle unique trajectory IDs
    traj_ids = list(traj_lengths.keys())
    rng.shuffle(traj_ids)
    
    # Select trajectories
    selected_ids = []
    rows_so_far = 0
    
    for traj_id in traj_ids:
        traj_rows = traj_lengths[traj_id]
        
        # If adding this trajectory would exceed target and we already have some data, stop
        if rows_so_far + traj_rows > target_rows and rows_so_far > 0:
            break
            
        selected_ids.append(traj_id)
        rows_so_far += traj_rows
        
    
    # Create subset dataframe with selected trajectories
    df_tiny = df[df["traj_id"].isin(selected_ids)].copy()
    
    return df_tiny


def split_df(_df, _rng, _train_part):
    traj_ids = _df["traj_id"].unique()
    _rng.shuffle(traj_ids)

    n = len(traj_ids)
    n_tr = int(_train_part*n)

    tr_ids = traj_ids[:n_tr]
    va_ids = traj_ids[n_tr:]

    # Then we map each traj_id to its assigned split and add a new "split" column to the full dataset, ensuring each trajectory belongs entirely to only one set (train/val/test).

    split_map = {tid: "train" for tid in tr_ids}
    split_map.update({tid: "val" for tid in va_ids})

    # df["split"] = df["traj_id"].map(split_map)
    _df["split"] = _df["traj_id"].map(split_map)


def k_fold_split(df, n_splits, rng):
    traj_ids = df["traj_id"].unique()
    rng.shuffle(traj_ids)
    
    n = len(traj_ids)
    fold_size = n // n_splits
    
    # Create mapping from traj_id to fold number
    split_map = {}
    for i in range(n_splits):
        start_idx = i * fold_size
        # Last fold gets any remaining trajectories
        end_idx = (i + 1) * fold_size if i < n_splits - 1 else n
        
        fold_traj_ids = traj_ids[start_idx:end_idx]
        for tid in fold_traj_ids:
            split_map[tid] = i
    
    df["split_num"] = df["traj_id"].map(split_map)
    return df


def cross_validate(df, model, n_splits=5):
    # Identify coordinate and velocity columns
    coord_cols = [c for c in df.columns if c.startswith(("x_", "y_"))]
    vel_cols = [c for c in df.columns if c.startswith(("v_x_", "v_y_"))]
    
    rmse_scores = []
    mae_scores = []
    
    for fold in range(n_splits):
        # Split data
        train_df = df[df["split_num"] != fold].copy()
        val_df = df[df["split_num"] == fold].copy()
        
        # Prepare features for training set
        for col in coord_cols:
            train_df[f"init_{col}"] = train_df.groupby("traj_id")[col].transform('first')
        for col in vel_cols:
            train_df[f"init_{col}"] = train_df.groupby("traj_id")[col].transform('first')
        
        # Prepare features for validation set
        for col in coord_cols:
            val_df[f"init_{col}"] = val_df.groupby("traj_id")[col].transform('first')
        for col in vel_cols:
            val_df[f"init_{col}"] = val_df.groupby("traj_id")[col].transform('first')
        
        # Build feature and target matrices
        init_coord_cols = [f"init_{c}" for c in coord_cols]
        
        X_train = train_df[init_coord_cols + ["t"]].values
        y_train = train_df[coord_cols].values
        X_val = val_df[init_coord_cols + ["t"]].values
        y_val = val_df[coord_cols].values
        
        # Train model
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = fold_model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Fold {fold + 1}/{n_splits} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Calculate overall statistics
    results = {
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rmse_scores": rmse_scores,
        "mae_scores": mae_scores
    }
    
    print(f"\nCross-Validation Results:")
    print(f"RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    
    return results


def prepare_data(df, split_name):
    # Choose split
    df_subset = df[df["split"] == split_name].copy()

    # Prepare features
    coord_cols = [c for c in df_subset.columns if c.startswith(("x_", "y_"))]
    vel_cols = [c for c in df_subset.columns if c.startswith(("v_x_", "v_y_"))]

    # Add initial coordinates for each trajectory
    for col in coord_cols:
        df_subset[f"init_{col}"] = df_subset.groupby("traj_id")[col].transform('first')
    
    # Add initial velocities for each trajectory
    for col in vel_cols:
        df_subset[f"init_{col}"] = df_subset.groupby("traj_id")[col].transform('first')
    
    # Column names for initial values
    init_coord_cols = [f"init_{c}" for c in coord_cols]
    init_vel_cols = [f"init_{c}" for c in vel_cols]
    
    # Build feature matrix X and target matrix y
    # X = df_subset[init_coord_cols + init_vel_cols + ["t"]].values
    X = df_subset[init_coord_cols + ["t"]].values
    y = df_subset[coord_cols].values
    
    return X, y


def load_and_prepare_data_test(_path):
    df = pd.read_csv(_path)

    coord_cols = [c for c in df.columns if c.startswith(("x0_", "y0_"))]
    
    # Prepare features
    coord_values = df[coord_cols].values
    time_values = df["t"].values.reshape(-1, 1)
    
    # Create zeros for velocity columns
    n_rows = len(df)
    velocity_zeros = np.zeros((n_rows, 6))
    
    # Concatenate columns
    # X = np.concatenate([coord_values, velocity_zeros, time_values], axis=1)
    X = np.concatenate([coord_values, time_values], axis=1)
    
    return X


def export_predictions_to_csv(y_pred, output_path):
    # Create dataframe with predictions
    pred_df = pd.DataFrame(y_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    
    # Add Id column (starting from 0)
    pred_df.insert(0, 'Id', range(len(pred_df)))
    
    # Save to CSV
    pred_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Shape: {pred_df.shape}")
    print("First few rows:")
    print(pred_df.head())
    
    return pred_df

rng = np.random.default_rng(12)

df_clean = load_process_and_clean_csv(TRAIN_PATH)
df_tiny = generate_df_subset(df_clean, 0.02, rng)
k_fold_data = k_fold_split(df_tiny, 5, rng)
split_df(df_tiny, rng, 0.75)

X_train, y_train = prepare_data(df_tiny, "train")
X_val, y_val = prepare_data(df_tiny, "val")
X_test = load_and_prepare_data_test(TEST_PATH)