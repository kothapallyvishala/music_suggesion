# Step 1: Install necessary libraries (if not installed)
!pip install pandas scikit-learn

# Step 2: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import files

# Upload CSV file manually
uploaded = files.upload()

# Step 3: Create a sample dataset of songs with features


df = pd.read_csv("music_data.csv")
# Step 4: Encode categorical data (genre & artist) into numerical values
encoder = LabelEncoder()
df["genre_encoded"] = encoder.fit_transform(df["genre"])
df["artist_encoded"] = encoder.fit_transform(df["artist"])

# Step 5: Select features for recommendation
features = df[["tempo", "danceability", "energy", "genre_encoded", "artist_encoded"]]

# Step 6: Train a KNN model
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(features)

# Step 7: Function to Recommend Similar Songs
def recommend_song(song_name):
    if song_name not in df["song"].values:
        return "Song not found in database!"

    song_index = df[df["song"] == song_name].index[0]
    distances, indices = knn.kneighbors([features.iloc[song_index]])

    recommended_songs = df.iloc[indices[0][1:]]["song"].tolist()
    return recommended_songs

# Step 8: Test the Recommendation System
user_input = input()
recommended_songs = recommend_song(user_input)
print(f"Because you liked '{user_input}', you may also like: {recommended_songs}")
