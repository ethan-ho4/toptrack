import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import load_data

def preprocess_data(df):
    """
    Select features, define target, and normalize data.
    """
    print("Preprocessing data...")
    
    # 1. Feature Selection
    # Standard audio features provided by Spotify
    feature_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms', 'year'
    ]
    
    # Check if columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = df[feature_columns].copy()
    
    # 2. Target Definition
    # Create binary "Hit" column (Popularity > 70)
    # Note: 70 is a high threshold, we can adjust if classes are too imbalanced
    df['is_hit'] = (df['popularity'] > 70).astype(int)
    y = df['is_hit']
    
    print(f"Target distribution (Hit=1, Not Hit=0):\n{y.value_counts()}")

    # 3. Data Normalization
    # Scale features to 0-1 range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Load raw data
        df = load_data()
        
        # Preprocess features and target
        X, y = preprocess_data(df)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print("\nData Engineering Success!")
        print(f"Training Data: {X_train.shape}")
        print(f"Testing Data: {X_test.shape}")
        
    except Exception as e:
        print(f"Error in Data Engineering: {e}")
