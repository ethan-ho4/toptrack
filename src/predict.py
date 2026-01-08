import joblib
import pandas as pd
import os
import numpy as np

def load_prediction_artifacts():
    """
    Load the saved model and scaler.
    """
    model_path = os.path.join("models", "random_forest.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run evaluate.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run data_engineering.py first.")
        
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_song(features, model, scaler):
    """
    Predict if a song is a hit based on features.
    
    Args:
        features (dict): Dictionary containing song audio features.
        model: Trained classifier.
        scaler: Fitted scaler.
        
    Returns:
        dict: Prediction result and probability.
    """
    # Define the expected order of columns (must match training)
    feature_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms', 'year'
    ]
    
    # Create DataFrame from input features
    df = pd.DataFrame([features])
    
    # Ensure all columns exist (fill missing with 0 or handle error)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0 # Default value if missing
            
    # Reorder columns to match training
    df = df[feature_columns]
    
    # Scale features
    df_scaled_array = scaler.transform(df)
    # Convert back to DataFrame to preserve feature names for the model
    df_scaled = pd.DataFrame(df_scaled_array, columns=feature_columns)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1] # Probability of Class 1 (Hit)
    
    return {
        "is_hit": bool(prediction),
        "hit_probability": round(probability, 4)
    }

if __name__ == "__main__":
    try:
        model, scaler = load_prediction_artifacts()
        
        # Example: Input features for a hypothetical new song "Espresso" or similar trending track
        # Values are illustrative
        new_song = {
            'danceability': 0.70,
            'energy': 0.85,
            'key': 1,
            'loudness': -5.5,
            'mode': 1,
            'speechiness': 0.05,
            'acousticness': 0.10,
            'instrumentalness': 0.00,
            'liveness': 0.12,
            'valence': 0.85,
            'tempo': 120.0,
            'duration_ms': 200000,
            'year': 2024
        }
        
        print(f"\nAnalyzing Song Features: {new_song}")
        result = predict_song(new_song, model, scaler)
        
        print("\n" + "="*30)
        print(" PREDICTION RESULT ")
        print("="*30)
        
        if result['is_hit']:
            print(f"🔥 HIT! (Probability: {result['hit_probability']*100:.2f}%)")
        else:
            print(f"❄️ NOT A HIT. (Probability: {result['hit_probability']*100:.2f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
