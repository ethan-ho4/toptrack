import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.data_engineering import preprocess_data, split_data
import time

def get_models():
    """
    Returns a dictionary of initialized models.
    """
    return {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=20, eval_metric='logloss', use_label_encoder=False, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models.
    """
    
    # Initialize models
    models = get_models()
    
    results = {}
    
    print(f"\nTraining {len(models)} models on {X_train.shape[0]} samples...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        elapsed_time = time.time() - start_time
        print(f"Done in {elapsed_time:.2f}s | Accuracy: {accuracy:.4f}")
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report
        }

    return results

if __name__ == "__main__":
    try:
        # Load and Prepare Data
        df = load_data()
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train Models
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Detailed Reports
        print("\n" + "="*60)
        print("FINAL EVALUATION REPORTS")
        print("="*60)
        for name, data in results.items():
            print(f"\nModel: {name}")
            print(data['report'])
            
    except Exception as e:
        print(f"Error during training: {e}")
