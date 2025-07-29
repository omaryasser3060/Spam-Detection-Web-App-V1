# Spam Detection Web App

This is a simple machine learning web application that detects whether a message is **spam** or **ham (not spam)**. It uses a trained ML model and a lightweight UI built with Streamlit.

## 📌 Project Overview

- **Goal:** Classify SMS or email-like messages into spam or ham.
- **Model:** Trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **Interface:** Built using Python and Streamlit for fast testing and usability.

## ⚙️ Features

- Message classification (spam or ham)
- Clean and simple web UI
- Reusable saved model (`.pkl`) file
- Ready to extend for other languages, message types, or real-world data

## 🧠 Tech Stack

- Python 3
- Scikit-learn
- Pandas, NLTK
- Streamlit (for the UI)

## 📂 Project Structure
spam_application/
- ├── spam_detector.ipynb # Data processing and model training
- ├── spam_detector_model.pkl # Trained model
- ├── spam_application.py # Web UI with Streamlit
- ├── SMSSpam Collection # Dataset (TSV)

##Run the Streamlit app
- streamlit run spam_application.py


