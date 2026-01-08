import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from src.predict import load_prediction_artifacts, predict_song
from src.data_loader import load_data
import pandas as pd
import logging

# Silence Spotipy and request logging
logging.getLogger('spotipy').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv()

def get_spotify_client():
    """
    Initialize Spotify client.
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        return None
        
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager)

import sys
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr to silence Spotipy's noisy printer.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

import difflib

def search_local_data(df, query):
    """
    Search for a song in the local dataframe using smart matching and fuzzy logic.
    """
    query_str = str(query).lower().strip()
    
    # 1. Exact Substring Match (Fastest)
    matches = df[df['name'].str.lower().str.contains(query_str, na=False, regex=False)]
    
    # 2. Token Match
    if matches.empty:
        tokens = query_str.split()
        if len(tokens) > 1:
            mask = df.apply(lambda row: all(token in str(row['name']).lower() or token in str(row['artists']).lower() for token in tokens), axis=1)
            matches = df[mask]

    # 3. Fuzzy Search (Slowest but handles typos)
    # We only search the top 50,000 most popular songs to keep it fast
    if matches.empty:
        print("  (Performing fuzzy search for typos...)")
        # Get top 50k popular songs
        top_songs = df.sort_values(by='popularity', ascending=False).head(50000)
        # Create a dictionary map for fast lookup
        # We search primarily on NAME
        song_names = top_songs['name'].tolist()
        
        # Find closest match
        close_matches = difflib.get_close_matches(query_str, song_names, n=1, cutoff=0.6)
        
        if close_matches:
            best_guess_name = close_matches[0]
            print(f"  Did you mean: '{best_guess_name}'?")
            matches = df[df['name'] == best_guess_name]

    if matches.empty:
        raise ValueError(f"Could not find any song matching '{query}'.")
    
    # Sort by popularity to get the most likely match (the "hit" version)
    best_match = matches.sort_values(by='popularity', ascending=False).iloc[0]
    
    # Convert row to dict
    features = best_match.to_dict()
    
    # Clean up artist name
    artist_name = str(features['artists']).replace("['", "").replace("']", "").replace("'", "")
    
    return features, features['name'], artist_name

def get_track_features(sp, df, query):
    """
    Search for a song and retrieve its audio features.
    Tries API first, falls back to local data.
    """
    # 1. Try Spotify API
    if sp:
        try:
            # Suppress the noisy HTTP 403 print from spotipy
            with suppress_stderr():
                results = sp.search(q=query, limit=1, type='track')
                
                if results['tracks']['items']:
                    track = results['tracks']['items'][0]
                    track_id = track['id']
                    track_name = track['name']
                    artist_name = track['artists'][0]['name']
                    release_date = track['album']['release_date']
                    year = int(release_date.split('-')[0])
                    
                    # Try getting features
                    features = sp.audio_features(track_id)[0]
                    if features:
                        features['year'] = year
                        return features, track_name, artist_name
        except Exception:
             print("  (API unavailable. Switching to offline database...)")
    
    # 2. Fallback to Local Data
    return search_local_data(df, query)

def main():
    try:
        model, scaler = load_prediction_artifacts()
        
        # Load local data silently
        print("Initializing System...")
        df = load_data(verbose=False)
        sp = get_spotify_client()
        
        print("\n🎵 Spotify Hit Predictor Ready! 🎵")
        
        while True:
            query = input("\nEnter song name (or 'q' to quit): ")
            if query.lower() == 'q':
                break
                
            try:
                features, track_name, artist_name = get_track_features(sp, df, query)
                
                result = predict_song(features, model, scaler)
                
                print("-" * 40)
                print(f"🎤 {track_name} - {artist_name}")
                prob = result['hit_probability'] * 100
                
                if result['is_hit']:
                    print(f"🔥 PREDICTION: HIT ({prob:.1f}%)")
                else:
                    print(f"❄️ PREDICTION: FLOP ({prob:.1f}%)")
                print("-" * 40)
                    
            except Exception as e:
                # print(f"❌ {e}")
                print(f"❌ {e}")

    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
