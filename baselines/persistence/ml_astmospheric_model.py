#!/usr/bin/env python3
"""
Example script to demonstrate a machine-learning workflow that emulates PyMSIS-like functionality.

It:
1) Generates synthetic data for demonstration.
2) Merges (simulated) initial states, OMNI2, GOES, and density data into a single training DataFrame.
3) Performs basic feature engineering.
4) Trains an XGBoost regressor to predict atmospheric density.
5) Shows how to define an inference function that returns predicted density given (time, lat, lon, alt).

Usage:
  python storm_ai_pymsis_example.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

###############################################################################
# 1. Generate / Load Synthetic Data
###############################################################################

def generate_synthetic_initial_states(num_samples=1000, start_date="2025-01-01"):
    """
    Creates a synthetic DataFrame that mimics the 'initial_states.csv' format.
    """
    # Generate some random times
    start_dt = datetime.fromisoformat(start_date)
    timestamps = [start_dt + timedelta(hours=i*12) for i in range(num_samples)]
    
    df = pd.DataFrame({
        "File ID": np.arange(num_samples).astype(str).tolist(),
        "Timestamp": timestamps,
        "Semi-major Axis (km)": np.random.uniform(6500, 7100, num_samples),
        "Eccentricity": np.random.uniform(0, 0.01, num_samples),
        "Inclination (deg)": np.random.uniform(0, 90, num_samples),
        "RAAN (deg)": np.random.uniform(0, 360, num_samples),
        "Argument of Perigee (deg)": np.random.uniform(0, 360, num_samples),
        "True Anomaly (deg)": np.random.uniform(0, 360, num_samples),
        "Latitude (deg)": np.random.uniform(-90, 90, num_samples),
        "Longitude (deg)": np.random.uniform(-180, 180, num_samples),
        "Altitude (km)": np.random.uniform(200, 800, num_samples),
    })
    return df

def generate_synthetic_omni2(num_samples=1000):
    """
    Creates a synthetic OMNI2-like DataFrame with random space-weather values.
    """
    # For simplicity, let's assume each sample is already time-aligned with the initial states
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=num_samples, freq='12H'),
        "Kp_index": np.random.uniform(0, 9, num_samples),
        "f10.7_index": np.random.uniform(70, 200, num_samples),
        "Dst_index_nT": np.random.uniform(-100, 20, num_samples),
        "SW_Proton_Density_N_cm3": np.random.uniform(0, 20, num_samples),
    })
    return df

def generate_synthetic_goes(num_samples=1000):
    """
    Creates a synthetic GOES-like DataFrame with random X-ray flux values.
    """
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=num_samples, freq='12H'),
        "xrsa_flux": np.random.uniform(1e-9, 1e-7, num_samples),
        "xrsb_flux": np.random.uniform(1e-9, 1e-7, num_samples),
        "xrsa_flag": np.random.choice([0.0, 1.0, 2.0], size=num_samples),
        "xrsb_flag": np.random.choice([0.0, 1.0, 2.0], size=num_samples),
    })
    return df

def generate_synthetic_density(num_samples=1000):
    """
    Creates a synthetic density DataFrame with random orbit-mean densities.
    """
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=num_samples, freq='12H'),
        "Orbit Mean Density (kg/m^3)": np.random.uniform(1e-12, 1e-10, num_samples),
    })
    return df

###############################################################################
# 2. Merge Data
###############################################################################

def merge_data_on_timestamp(init_df, omni2_df, goes_df, density_df):
    """
    In a real scenario, you'd carefully align by 'File ID' and time range.
    Here, we just merge on Timestamp (or time).
    """
    # Rename columns in GOES to match
    goes_renamed = goes_df.rename(columns={"time": "Timestamp"})

    # Merge everything on 'Timestamp'
    merged = init_df.merge(omni2_df, on="Timestamp", how="left")
    merged = merged.merge(goes_renamed, on="Timestamp", how="left")
    merged = merged.merge(density_df, on="Timestamp", how="left")

    return merged

###############################################################################
# 3. Feature Engineering
###############################################################################

def generate_time_features(df, time_col="Timestamp"):
    """
    Example time-based feature engineering.
    """
    df["Hour"] = df[time_col].dt.hour
    df["DayOfYear"] = df[time_col].dt.dayofyear
    
    # Convert Timestamp to float (days since start)
    df["TimestampNum"] = (df[time_col] - df[time_col].min()).dt.total_seconds() / 86400.0

    return df

def generate_spatial_features(df, lat_col="Latitude (deg)", lon_col="Longitude (deg)"):
    """
    Example to transform lat/lon to radians or do something else.
    """
    df["Lat_rad"] = np.radians(df[lat_col])
    df["Lon_rad"] = np.radians(df[lon_col])
    return df

###############################################################################
# 4. Train-Test Split, Model Training, and Evaluation
###############################################################################

def main():
    # Generate synthetic data
    initial_states = generate_synthetic_initial_states(num_samples=1000)
    omni2_data = generate_synthetic_omni2(num_samples=1000)
    goes_data = generate_synthetic_goes(num_samples=1000)
    density_data = generate_synthetic_density(num_samples=1000)

    # Merge data
    training_df = merge_data_on_timestamp(
        initial_states, omni2_data, goes_data, density_data
    )

    # Basic cleanup: drop rows missing the target
    training_df.dropna(subset=["Orbit Mean Density (kg/m^3)"], inplace=True)

    # Feature engineering
    training_df = generate_time_features(training_df, time_col="Timestamp")
    training_df = generate_spatial_features(training_df, 
                                            lat_col="Latitude (deg)", 
                                            lon_col="Longitude (deg)")

    # Define feature columns (example subset)
    feature_cols = [
        "Semi-major Axis (km)", 
        "Inclination (deg)", 
        "RAAN (deg)",
        "Argument of Perigee (deg)",
        "True Anomaly (deg)",
        "Latitude (deg)",
        "Longitude (deg)",
        "Altitude (km)",
        "Kp_index",
        "f10.7_index",
        "Dst_index_nT",
        "SW_Proton_Density_N_cm3",
        "xrsa_flux",
        "xrsb_flux",
        "Hour",
        "DayOfYear",
        "TimestampNum",
        "Lat_rad",
        "Lon_rad"
    ]
    
    # Fill any remaining NaN with some default (e.g., mean or zero)
    for col in feature_cols:
        if col not in training_df.columns:
            training_df[col] = 0.0
        training_df[col].fillna(training_df[col].mean(), inplace=True)
    
    # Prepare data
    X = training_df[feature_cols]
    y = training_df["Orbit Mean Density (kg/m^3)"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Train model (XGB as example)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4e} kg/m^3")

    # Optionally, store model or proceed to inference usage
    # e.g., model.save_model("density_model.json")

    # Let's do a quick example inference call
    inference_time = X_test.iloc[0]["TimestampNum"]  # just demonstration
    test_row = X_test.iloc[0:1].copy()
    predicted_density = model.predict(test_row)
    print(f"Example predicted density at row 0: {predicted_density[0]:.4e} kg/m^3")

    # Demonstrate the 'get_density' approach:
    # (in real usage, we'd need actual space-weather for new times, lat/lon/alt, etc.)
    time_example = datetime(2025, 1, 20, 12, 0, 0)
    lat_example = 35.0
    lon_example = -120.0
    alt_example = 400.0
    density_estimate = get_density(time_example, lat_example, lon_example, alt_example, 
                                   model, training_df, feature_cols)
    print(f"PyMSIS-like estimate for {time_example}, lat={lat_example}, lon={lon_example}, alt={alt_example} km:")
    print(f"  => {density_estimate:.4e} kg/m^3")

def get_density(time_dt, lat, lon, alt, model, reference_df, feature_cols):
    """
    Emulates a PyMSIS-like function call.
    1) Construct feature vector for the given time + lat/lon/alt + relevant space-weather parameters.
    2) Use the trained model to predict density.
    
    *In a real system*, you'd fetch the correct space-weather data for 'time_dt'
    from your repository or real-time feed. Here we just approximate by:
      - Taking the row from `reference_df` that is closest in time
      - Overwriting lat/lon/alt with the requested ones
      - Using the model to predict.

    Args:
      time_dt (datetime): The requested date/time for density prediction.
      lat, lon, alt (float): Geodetic coordinates in degrees, degrees, km.
      model: A trained regression model with predict method.
      reference_df (pd.DataFrame): DataFrame containing time + space-weather to sample from.
      feature_cols (list): The columns the model expects.

    Returns:
      float: The predicted density in kg/m^3
    """
    # Find the row in reference_df with Timestamp closest to time_dt
    # For simplicity, we’ll do a nearest approach:
    reference_df["diff_time"] = reference_df["Timestamp"].apply(lambda x: abs((x - time_dt).total_seconds()))
    nearest_row_idx = reference_df["diff_time"].idxmin()
    row = reference_df.loc[nearest_row_idx, :].copy()

    # Overwrite lat/lon/alt
    row["Latitude (deg)"] = lat
    row["Longitude (deg)"] = lon
    row["Altitude (km)"] = alt

    # Re-engineer the features that might change (time-based, lat/lon rad, etc.)
    row["Hour"] = time_dt.hour
    row["DayOfYear"] = time_dt.timetuple().tm_yday
    timestamp_num = (time_dt - reference_df["Timestamp"].min()).total_seconds() / 86400.0
    row["TimestampNum"] = timestamp_num
    row["Lat_rad"] = np.radians(lat)
    row["Lon_rad"] = np.radians(lon)

    # Model expects a DF row
    row_df = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    density_pred = model.predict(row_df)[0]

    # Clean up any helper columns from reference_df
    reference_df.drop(columns=["diff_time"], inplace=True)

    return float(density_pred)

# Entry point
if __name__ == "__main__":
    main()
