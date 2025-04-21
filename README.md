# Real-Time-Fraud-Detection
A real-time fraud detection pipeline that combines Apache Kafka, Apache Flink, and a PyTorch-based Autoencoder model to detect anomalous financial transactions. Built for streaming environments with real-time analysis and alerting.
🚀 Features

✅ Real-time streaming with Kafka

🧠 Anomaly detection using Autoencoder (PyTorch)

🔄 Scalable stream processing via Apache Flink

🛆 Fully containerized using Docker

📡 Synthetic transaction data simulation

🔍 Fraud detection based on reconstruction error


🧪 How It Works

1. Generate Data: Simulated transactions are sent to Kafka.

2. Stream Processing: Flink consumes the stream and preprocesses each record.

3. Anomaly Detection: The trained autoencoder reconstructs each input; a high reconstruction error indicates potential fraud.

4. Alerting: Anomalous transactions are flagged for review.
