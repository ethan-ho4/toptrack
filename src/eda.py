import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.data_loader import load_data

def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the dataset.
    """
    print("Performing EDA...")
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation Matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Audio Features")
    plt.tight_layout()
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    correlation_plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_plot_path)
    print(f"\nCorrelation matrix saved to {correlation_plot_path}")
    
    # Distribution of Popularity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['popularity'], bins=30, kde=True)
    plt.title("Distribution of Song Popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Count")
    plt.tight_layout()
    
    popularity_plot_path = os.path.join(output_dir, "popularity_distribution.png")
    plt.savefig(popularity_plot_path)
    print(f"Popularity distribution saved to {popularity_plot_path}")

    # Scatter plot: Loudness vs Popularity (as per roadmap)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='loudness', y='popularity', data=df, alpha=0.1)
    plt.title("Loudness vs Popularity")
    plt.xlabel("Loudness")
    plt.ylabel("Popularity")
    plt.tight_layout()
    
    loudness_plot_path = os.path.join(output_dir, "loudness_vs_popularity.png")
    plt.savefig(loudness_plot_path)
    print(f"Loudness vs Popularity plot saved to {loudness_plot_path}")

if __name__ == "__main__":
    try:
        df = load_data()
        perform_eda(df)
    except Exception as e:
        print(f"Error during EDA: {e}")
