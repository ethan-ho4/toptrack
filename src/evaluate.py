import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.data_loader import load_data
from src.data_engineering import preprocess_data
from src.train_models import get_models

def run_evaluation():
    """
    Runs Cross-Validation and generates a Leaderboard.
    """
    print("Starting Comparative Analysis & Evaluation (Phase 4)...")
    
    # Load Data
    df = load_data()
    X, y = preprocess_data(df)
    
    # Initialize Models
    models = get_models()
    
    # Configuration for Cross-Validation
    # Using StratifiedKFold because of the class imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    print(f"\nRunning {cv.get_n_splits()}-Fold Cross-Validation on {len(models)} models...")
    print("This might take a few minutes...\n")
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        # Using 'f1' or 'roc_auc' is better for imbalance, but we'll stick to accuracy as per roadmap 
        # (or maybe f1 to be smarter). Let's do Accuracy as requested but maybe print others.
        # The roadmap explicitly asked for "Metric Calculation" but also "Leaderboard... comparing accuracy".
        # I will calculate Accuracy for the chart.
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"  -> Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        results[name] = mean_score
        
        # Train and Save the final model on full data
        print(f"  -> Training final {name} on full dataset...")
        model.fit(X, y)
        
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"  -> Saved to {model_path}")

    # Generate Leaderboard Visualization
    print("\nGenerating Leaderboard Chart...")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.title("Model Accuracy Leaderboard (5-Fold CV)")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.8, 1.0) # Zoom in to see differences if they are high
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "model_leaderboard.png")
    plt.savefig(plot_path)
    print(f"Leaderboard saved to {plot_path}")
    
    # Determine Winner
    winner = max(results, key=results.get)
    print(f"\n🏆 The Winning Model is: {winner} with {results[winner]:.4f} accuracy!")
    print(f"You can find the saved model in models/{winner.replace(' ', '_').lower()}.pkl")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Error during evaluation: {e}")
