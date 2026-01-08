# 🎵 Spotify Hit Predictor

**Spotify Hit Predictor** is a machine learning project that analyzes audio features of songs (like danceability, energy, and tempo) to predict whether they will be a "Hit". 

The project compares four different classification models—**Logistic Regression, Random Forest, XGBoost, and KNN**—to find the most accurate predictor. It also includes a **Live Predictor** tool that lets you search for *any* song (via Spotify API or local database) and check its hit potential in real-time.

## 🚀 Features

*   **Multi-Model Analysis**: Trains and evaluates 4 distinct ML models.
*   **Comprehensive EDA**: Generates visualizations for feature correlations and popularity distribution.
*   **Live Prediction Tool**: Interactive CLI tool to predict hits for new songs.
*   **Hybrid Search System**: 
    *   Connects to **Spotify API** for real-time data.
    *   Falls back to a **Local Database (170k+ songs)** if API is unavailable.
    *   Includes **Fuzzy Search** to handle typos in song names.
*   **Robust Pipeline**: Modular code for Data Loading, Engineering, Training, and Evaluation.

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ethan-ho4/toptrack.git
    cd toptrack
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Setup

### 1. Data Setup (Automated)
The project is designed to automatically download the [Spotify 160k+ Dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks) from Kaggle.
*   Simply run any script (like `src/data_loader.py`), and it will fetch the data into a `data/` folder if it's missing.

### 2. Spotify API Setup (Optional but Recommended)
To enable real-time searching of *new* songs (post-2021) using the Spotify API:

1.  Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2.  Create an App to get your **Client ID** and **Client Secret**.
3.  Create a `.env` file in the project root:
    ```ini
    SPOTIPY_CLIENT_ID="your_client_id_here"
    SPOTIPY_CLIENT_SECRET="your_client_secret_here"
    ```
*(Note: If you skip this, the tool will strictly use the offline database)*

---

## 🏃 Usage

### 🔮 Live Predictor (The Main Tool)
Interact with the trained model to predict hits.
```bash
python main.py
```
*   **Input**: Type any song name (e.g., "Blinding Lights").
*   **Output**: The model's prediction (🔥 HIT or ❄️ FLOP) and the probability score.

### 🏗️ Re-running the Pipeline
If you want to retrain models or see the analysis from scratch:

1.  **Exploratory Data Analysis (EDA)**:
    Generates plots in `plots/` folder.
    ```bash
    python -m src.eda
    ```

2.  **Data Engineering**:
    Prepares features and splits data. Saved scaler to `models/scaler.pkl`.
    ```bash
    python -m src.data_engineering
    ```

3.  **Train & Evaluate Models**:
    Trains all models, runs Cross-Validation, and saves the leaderboard.
    ```bash
    python -m src.evaluate
    ```
    *   *Result*: Saves the best model to `models/random_forest.pkl`.

---

## 📊 Model Performance

After rigorous 5-Fold Cross-Validation, the models performed as follows:

| Rank | Model | Accuracy |
| :--- | :--- | :--- |
| 🥇 | **Random Forest** | **~97.7%** |
| 🥈 | K-Nearest Neighbors | ~97.6% |
| 🥉 | XGBoost | ~94.8% |
| 4th | Logistic Regression | ~77.4% |

*The **Random Forest** model is currently used for all live predictions.*

## 📂 Project Structure
```
toptrack/
├── data/                   # Dataset (auto-downloaded)
├── models/                 # Saved models (.pkl)
├── plots/                  # EDA and Evaluation charts
├── src/
│   ├── data_loader.py      # Handles data ingestion
│   ├── eda.py              # Exploratory Data Analysis
│   ├── data_engineering.py # Cleaning & Feature Scaling
│   ├── train_models.py     # Model Training Definitions
│   ├── evaluate.py         # Cross-Validation & Leaderboard
│   ├── predict.py          # Prediction Logic
│   └── live_predict.py     # CLI Application
├── requirements.txt        # Python dependencies
└── README.md               # Project Documentation
```
