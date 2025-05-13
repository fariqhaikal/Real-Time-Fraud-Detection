import pandas as pd
import numpy as np
import joblib

# Function to preprocess transaction data (same as in kafka_fraud_detector)
def preprocess(transaction, encoder, scaler):
    df = pd.DataFrame([transaction])

    # Calculate Balance After
    df["Balance After"] = df["Balance Before"] - df["Transaction Amount"]

    # Extract Device ID (convert to numeric)
    df["Device ID"] = df["Device ID"].str.extract(r"(\d+)").astype(float)

    # Split IP Address into 4 columns
    df[["IP1", "IP2", "IP3", "IP4"]] = df["IP Address"].str.split('.', expand=True).astype(float)
    df = df.drop(columns=["IP Address"])

    # Convert Timestamp to float
    df["Timestamp"] = df["Timestamp"].apply(lambda x: float(x))

    # Process "Age of Account"
    df["Age of Account"] = df["Age of Account"].str.extract(r"(\d+)").astype(float)

    # One-hot encode categorical features
    categorical = ["Location", "Merchant Category", "Transaction Type", "Payment Method", "Country Code"]
    encoded = encoder.transform(df[categorical])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical))

    df = df.drop(columns=categorical)
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Scale features
    scaled = scaler.transform(df)
    return scaled
