import os
import csv
import json
import torch
import numpy as np
from kafka import KafkaConsumer
import joblib
from autoencoder_model import Autoencoder
from utils import preprocess

# Create output directory
os.makedirs("Flag", exist_ok=True)

# Load model and preprocessors
autoencoder = Autoencoder(input_size=55)
autoencoder.load_state_dict(torch.load("model/autoencoder.pth"))
autoencoder.eval()

scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/one_hot_encoder.pkl")

# Kafka consumer setup
consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

def log_transaction(filename, customer_id, reasons):
    filepath = os.path.join("Flag", filename)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Customer ID",
                "High Transaction Amount",
                "Unusual Location",
                "Suspicious Payment Method",
                "Account Age < 30 Days",
                "High Reconstruction Loss"
            ])
        # Ensure the list of reasons has the correct length and fill missing reasons with ""
        padded_reasons = reasons + [""] * 5
        writer.writerow([customer_id] + padded_reasons[:5])

# Real-time detection loop
for message in consumer:
    transaction = message.value
    try:
        reasons = []
        customer_id = transaction.get("Customer ID", "UNKNOWN")

        # Heuristic checks
        if transaction.get("Transaction Amount", 0) > 5000:
            reasons.append("High Transaction Amount")
        else:
            reasons.append("")

        if transaction.get("Location") not in ["Kuala Lumpur", "Johor Bahru", "Penang"]:
            reasons.append("Unusual Location")
        else:
            reasons.append("")

        if transaction.get("Payment Method") == "Cryptocurrency":
            reasons.append("Suspicious Payment Method")
        else:
            reasons.append("")

        # Parse 'Age of Account' and check if it's less than 30 days
        age_str = transaction.get("Age of Account", "0 days")
        try:
            age_days = int(age_str.split()[0])
            if age_days < 30:
                reasons.append("Account Age < 30 Days")
            else:
                reasons.append("")
        except:
            reasons.append("")  # In case parsing fails

        # Preprocess and compute loss
        x = preprocess(transaction, encoder, scaler)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = autoencoder(x_tensor)
            loss = torch.nn.functional.mse_loss(reconstructed, x_tensor)

        loss_value = loss.item()
        if loss_value > 0.15:
            reasons.append(f"High Reconstruction Loss: {loss_value:.4f}")
        else:
            reasons.append("")

        # Detect Fraud and log transaction
        if loss_value > 0.15 or len([reason for reason in reasons if reason != ""]) > 0:
            log_transaction("fraud.csv", customer_id, reasons)
            print(f"🚨 FRAUD DETECTED: Loss={loss_value:.4f} | Reasons: {reasons} | Transaction: {transaction}")
        else:
            log_transaction("normal.csv", customer_id, reasons)
            print(f"✅ Normal Transaction: Loss={loss_value:.4f} | Transaction: {transaction}")

    except Exception as e:
        print(f"⚠️ Skipped a message due to error: {e}")
