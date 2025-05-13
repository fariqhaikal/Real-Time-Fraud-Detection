from confluent_kafka import Producer
import json
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"

# Initialize Kafka Producer
producer = Producer({'bootstrap.servers': KAFKA_BROKER})

# Function to generate normal & anomalous transactions
def generate_synthetic_data():
    anomaly_ratio = 0.05
    num_samples = 1000
    num_anomalies = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomalies
    
    # Generate normal transactions
    normal_data = {
        "Transaction Amount": np.round(np.random.uniform(10, 1000, num_normal), 2),
        "Timestamp": [datetime.now() - timedelta(minutes=random.randint(0, 100000)) for _ in range(num_normal)],
        "Location": np.random.choice(["Kuala Lumpur", "Penang", "Johor Bahru", "Kuching"], num_normal),
        "Merchant Category": np.random.choice(["Food", "Electronics", "Clothing", "Travel", "Groceries"], num_normal),
        "Customer ID": np.random.randint(10000, 99999, num_normal),
        "Device ID": [f"dev{random.randint(100, 999)}" for _ in range(num_normal)],
        "IP Address": [f"{random.randint(100, 255)}.{random.randint(100, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(num_normal)],
        "Transaction Type": np.random.choice(["Payment", "Refund", "Withdrawal"], num_normal),
        "Previous Transaction Amount": np.round(np.random.uniform(5, 950, num_normal), 2),
        "Velocity": np.random.randint(1, 10, num_normal),
        "Balance Before": np.round(np.random.uniform(100, 5000, num_normal), 2),
        "Payment Method": np.random.choice(["Credit Card", "Debit Card", "E-Wallet", "Bank Transfer"], num_normal),
        "Country Code": np.random.choice(["MY", "SG", "ID", "TH"], num_normal),
        "Session Duration": np.random.randint(30, 300, num_normal),
        "Age of Account": np.random.randint(1, 365 * 5, num_normal),  # in days
        "Transaction Frequency": np.random.randint(1, 20, num_normal),
    }
    
    # Generate anomalous transactions
    anomalous_data = {
        "Transaction Amount": np.round(np.random.uniform(1000, 10000, num_anomalies), 2),  # High values
        "Timestamp": [datetime.now() - timedelta(minutes=random.randint(0, 100000)) for _ in range(num_anomalies)],
        "Location": np.random.choice(["New York", "London", "Tokyo"], num_anomalies),  # Unusual locations
        "Merchant Category": np.random.choice(["Luxury", "Jewelry", "Gaming"], num_anomalies),  # Rare categories
        "Customer ID": np.random.randint(10000, 99999, num_anomalies),
        "Device ID": [f"dev{random.randint(1000, 9999)}" for _ in range(num_anomalies)],
        "IP Address": [f"{random.randint(1, 10)}.{random.randint(1, 10)}.{random.randint(1, 10)}.{random.randint(1, 10)}" for _ in range(num_anomalies)],  # Rare IPs
        "Transaction Type": np.random.choice(["Payment", "Transfer"], num_anomalies),
        "Previous Transaction Amount": np.round(np.random.uniform(900, 10000, num_anomalies), 2),
        "Velocity": np.random.randint(10, 50, num_anomalies),  # High frequency
        "Balance Before": np.round(np.random.uniform(5000, 10000, num_anomalies), 2),
        "Payment Method": np.random.choice(["Cryptocurrency", "Wire Transfer"], num_anomalies),  # Rare methods
        "Country Code": np.random.choice(["US", "UK", "JP"], num_anomalies),
        "Session Duration": np.random.randint(5, 50, num_anomalies),  # Shorter session
        "Age of Account": np.random.randint(1, 60, num_anomalies),  # Newer accounts
        "Transaction Frequency": np.random.randint(20, 50, num_anomalies),  # High frequency
    }
    
    normal_df = pd.DataFrame(normal_data)
    anomalous_df = pd.DataFrame(anomalous_data)
    
    # Calculate Balance After for both normal and anomalous data
    normal_df["Balance After"] = normal_df["Balance Before"] - normal_df["Transaction Amount"]
    anomalous_df["Balance After"] = anomalous_df["Balance Before"] - anomalous_df["Transaction Amount"]
    
    # Format Age of Account as days
    normal_df["Age of Account"] = normal_df["Age of Account"].apply(lambda x: f"{x} days")
    anomalous_df["Age of Account"] = anomalous_df["Age of Account"].apply(lambda x: f"{x} days")
    
    # Combine and shuffle data
    combined_df = pd.concat([normal_df, anomalous_df]).sample(frac=1).reset_index(drop=True)
    
    return combined_df

# Stream transactions to Kafka
while True:
    df = generate_synthetic_data()
    for _, transaction in df.iterrows():
        transaction_dict = transaction.to_dict()
        transaction_dict["Timestamp"] = transaction_dict["Timestamp"].timestamp()  # Convert to Unix time
        producer.produce(TOPIC, key=str(transaction_dict["Timestamp"]), value=json.dumps(transaction_dict))
        producer.flush()
        print(f"🔄 Sent Transaction: {transaction_dict}")
        time.sleep(1)  # Simulating real-time transactions
