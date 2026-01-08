import pandas as pd
import os
import kagglehub
import shutil

def download_data(target_dir="data"):
    """
    Downloads the Spotify dataset using kagglehub and moves it to the target directory.
    Returns the path to the CSV file.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("fcpercival/160k-spotify-songs-sorted")
    print("Path to dataset files:", path)

    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Find the CSV file in the downloaded path
    csv_file = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break
    
    if not csv_file:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    # Move/Copy to target directory
    target_path = os.path.join(target_dir, "spotify_data.csv")
    shutil.copy(csv_file, target_path)
    print(f"Dataset copied to {target_path}")
    return target_path

def load_data(filepath=None, verbose=True):
    """
    Load data from a CSV file. If filepath is not provided, it attempts to download/find it.
    """
    if filepath is None:
        filepath = os.path.join("data", "spotify_data.csv")
        if not os.path.exists(filepath):
            filepath = download_data()
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    df = pd.read_csv(filepath)
    if verbose:
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    # Example usage
    try:
        df = load_data()
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")