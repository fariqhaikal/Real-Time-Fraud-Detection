# 💳 Real-Time Fraud Detection Pipeline

A real-time fraud detection system built with **Apache Kafka**, **Apache Flink**, and a **PyTorch-based Autoencoder**, designed to detect suspicious financial transactions as they stream in. This project is fully containerized with Docker and is built to scale for production-like environments.

---

## 🚀 Features

- ✅ **Real-time streaming** with Kafka
- 🧠 **Deep Learning-based anomaly detection** (Autoencoder using PyTorch)
- 🔄 **Scalable stream processing** via Apache Flink
- 🐳 **Dockerized architecture** for easy setup
- 🧪 **Synthetic transaction data generation** with labeled anomalies
- 🔍 **Fraud detection logic** based on reconstruction error thresholding

---

## 🧪 How It Works

1. **Data Simulation**  
   A Python script generates synthetic normal and anomalous transaction data and streams it into Kafka.

2. **Stream Processing with Flink**  
   Flink consumes transactions from Kafka in real time, preparing them for analysis.

3. **Anomaly Detection**  
   A trained autoencoder tries to reconstruct each transaction. Large reconstruction errors signal anomalies.

4. **Fraud Alerting**  
   Anomalous transactions are printed or flagged for review in real time.

---

## 📦 Tech Stack

- **Kafka** – Distributed event streaming
- **Flink** – Real-time stream processing
- **PyTorch** – Deep learning (Autoencoder)
- **Docker Compose** – Container orchestration
- **Pandas / NumPy** – Data preprocessing
- **Confluent Kafka** – Kafka client for Python

---

## 🛠 Setup Instructions

```bash
# 1. Clone this repo
git clone https://gitlab.com/your-username/Real-Time-Fraud-Detection.git
cd Real-Time-Fraud-Detection

# 2. Start the services
docker-compose up -d

# 3. Train the model (generates autoencoder.pth, scaler.pkl, encoder.pkl)
python train_autoencoder.py

# 4. Start the producer (sends transactions to Kafka)
python kafka_producer.py

# 5. Start the Flink pipeline
python flink_consumer.py

# 6. (Optional) Run live fraud detection consumer
python fraud_detector.py
