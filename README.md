# 💳 Real-Time Fraud Detection System

A **real-time fraud detection system** designed to detect suspicious financial transactions as they stream in. Built with **Apache Kafka**, **PyTorch-based Autoencoder**, and a variety of heuristic checks, this system flags anomalous transactions for further review. This project is designed to be scalable, containerized with **Docker**, and tailored for production environments.

---

## 🚀 Features

- ✅ **Real-time streaming** with **Kafka**
- 🧠 **Deep learning-based anomaly detection** using **Autoencoders** (PyTorch)
- 🔍 **Fraud detection logic** based on both **reconstruction loss** and **heuristic checks** (e.g., high transaction amounts, unusual locations, suspicious payment methods, account age)
- 📦 **Real-time transaction flagging** into **CSV** files (`fraud.csv`, `normal.csv`)
- 🐳 **Dockerized architecture** for easy setup and deployment
- 💡 **Fraud detection insights** with **reasoning** for flagging suspicious transactions

---

## 🧪 How It Works

1. **Transaction Data Simulation**  
   A Python script generates synthetic transaction data, including both normal and anomalous transactions, which are then streamed into **Kafka**.

2. **Fraud Detection Consumer**  
   A Kafka consumer continuously processes incoming transactions:
   - **Deep Learning Model**: The autoencoder reconstructs transactions and flags those with large reconstruction errors.
   - **Heuristic Rules**: Transactions are also checked for suspicious behavior (e.g., high transaction amounts, unusual locations, and payment methods, and account age).
   
3. **Anomaly Detection**  
   If the reconstruction error exceeds a defined threshold or if heuristic checks fail, the transaction is flagged as potentially fraudulent.

4. **Real-time Flagging**  
   Flagged transactions are saved into `fraud.csv` with customer ID and reasons for flagging. Normal transactions are saved into `normal.csv`.

---

## 📦 Tech Stack

- **Kafka** – Distributed event streaming platform
- **PyTorch** – Deep learning framework (Autoencoder for anomaly detection)
- **Docker** – Containerization for deployment
- **Python** – Backend and data processing (using libraries such as Pandas, NumPy)
- **joblib** – Saving and loading model and preprocessors
- **Kafka Consumer** – Real-time processing of transaction streams

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Real-Time-Fraud-Detection.git
cd Real-Time-Fraud-Detection
