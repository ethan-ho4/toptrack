import os
from src.data_loader import load_data

def main():
    print("Spotify Hit Predictor - Started")
    
    # Placeholder for data path
    # You will need to download the dataset to the 'data' folder
    data_path = os.path.join("data", "dataset.csv")
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        print(df.head())
    else:
        print(f"Data file not found at {data_path}. Please download the dataset.")

if __name__ == "__main__":
    main()
