# Spotify Hit Predictor - Multi-Model Comparison

## Phase 1: Environment & Data Acquisition
- [x] **Set up Development Environment**: Initialize a Python script structure (VS Code) or Jupyter Notebook environment; install pandas, scikit-learn, spotipy, and seaborn.
- [x] **Data Sourcing**: Download the Spotify 160k+ dataset from Kaggle or configure Spotify API credentials to pull your own song library.
- [x] **Initial Data Inspection**: Perform Exploratory Data Analysis (EDA) to find correlations between audio features (e.g., does "Loudness" correlate with "Popularity"?).

## Phase 2: Data Engineering (The "Cleaning" Stage)
- [x] **Feature Selection**: Filter columns to focus on musical attributes: danceability, energy, tempo, valence, etc.
- [x] **Target Definition**: Create a binary "Hit" column (e.g., if popularity > 70, then 1, else 0).
- [x] **Data Normalization**: Apply StandardScaler or MinMaxScaler to ensure all features are on a scale of 0 to 1.
- [x] **Data Splitting**: Execute a Train-Test split (80% training / 20% testing).

## Phase 3: Model Development & Training
- [x] **Baseline Model**: Implement Logistic Regression to establish a minimum performance metric.
- [x] **Tree-Based Models**: Train a Random Forest Classifier and an XGBoost model to capture complex, non-linear patterns.
- [x] **Similarity Model**: Implement K-Nearest Neighbors (KNN) to see if "similar" sounding songs share popularity levels.

## Phase 4: Comparative Analysis & Evaluation
- [x] **Metric Calculation**: Compute Accuracy, Precision, Recall, and F1-Score for every model.
- [x] **Cross-Validation**: Run K-Fold Cross-Validation to ensure the models aren't just "getting lucky" on certain data.
- [x] **Leaderboard Generation**: Create a visualization (Bar Chart) comparing the accuracy of all four models.

## Phase 5: Testing & Deployment
- [ ] **Predict New Songs**: Input a currently trending song's audio features into the winning model to see if it predicts a "Hit."
- [ ] **Final Documentation**: Summarize which features were most important (e.g., "In 2024, Energy was a higher predictor than Tempo").
