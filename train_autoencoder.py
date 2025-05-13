import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime, timedelta
import random

# Function to generate synthetic data
def generate_synthetic_data(num_samples=500000):
    # Generate normal transactions
    normal_data = {
        "Transaction Amount": np.round(np.random.uniform(10, 1000, num_samples), 2),
        "Timestamp": [datetime.now() - timedelta(minutes=random.randint(0, 100000)) for _ in range(num_samples)],
        "Location": np.random.choice(["Kuala Lumpur", "Penang", "Johor Bahru", "Kuching"], num_samples),
        "Merchant Category": np.random.choice(["Food", "Electronics", "Clothing", "Travel", "Groceries"], num_samples),
        "Customer ID": np.random.randint(10000, 99999, num_samples),
        "Device ID": [f"dev{random.randint(100, 999)}" for _ in range(num_samples)],
        "IP Address": [f"{random.randint(100, 255)}.{random.randint(100, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(num_samples)],
        "Transaction Type": np.random.choice(["Payment", "Refund", "Withdrawal"], num_samples),
        "Previous Transaction Amount": np.round(np.random.uniform(5, 950, num_samples), 2),
        "Velocity": np.random.randint(1, 10, num_samples),
        "Balance Before": np.round(np.random.uniform(100, 5000, num_samples), 2),
        "Payment Method": np.random.choice(["Credit Card", "Debit Card", "E-Wallet", "Bank Transfer"], num_samples),
        "Country Code": np.random.choice(["MY", "SG", "ID", "TH"], num_samples),
        "Session Duration": np.random.randint(30, 300, num_samples),
        "Age of Account": np.random.randint(1, 365 * 5, num_samples),
        "Transaction Frequency": np.random.randint(1, 20, num_samples),
    }
    
    normal_df = pd.DataFrame(normal_data)
    normal_df["Timestamp"] = normal_df["Timestamp"].apply(lambda x: x.timestamp())  # Convert timestamps to numeric
    print(normal_df.shape)
    return normal_df

# Generate training data
train_df = generate_synthetic_data()
train_df["Balance After"] = train_df["Balance Before"] - train_df["Transaction Amount"]

# Convert Device ID to numeric (Extracting the numeric part from "devXXX")
train_df["Device ID"] = train_df["Device ID"].str.extract(r'(\d+)').astype(float)

# Convert IP Address to four separate numeric columns
train_df[["IP1", "IP2", "IP3", "IP4"]] = train_df["IP Address"].str.split('.', expand=True).astype(float)

# Drop original IP Address column (since it's now numeric)
train_df = train_df.drop(columns=["IP Address"])

# One-hot encode categorical features
categorical_features = ["Location", "Merchant Category", "Transaction Type", "Payment Method", "Country Code"]
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = one_hot_encoder.fit_transform(train_df[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns and concatenate encoded ones
train_df = train_df.drop(columns=categorical_features)
train_df = pd.concat([train_df, encoded_categorical_df], axis=1)

# Normalize all features
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_df)
train_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model
input_size = train_tensor.shape[1]
model = Autoencoder(input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_tensor)
    loss = criterion(outputs, train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}")

print(f"Number of features after preprocessing: {train_tensor.shape[1]}")

# Create 'model/' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model and preprocessors
torch.save(model.state_dict(), "model/autoencoder.pth")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(one_hot_encoder, "model/one_hot_encoder.pkl")

print("✅ Training completed. Model, scaler, and encoder saved!")
