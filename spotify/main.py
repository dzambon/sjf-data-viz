import os
import pickle
from sjf_data_viz.spotify import *

# -----------------------------------------------------
# Collect some data from spotify and save them
# -----------------------------------------------------
# 1) Create a dictionary of playlist IDs like the following
playlist_ids = {
    "punk": "37i9dQZF1DX3LDIBRoaCDQ",
    "reggae": "37i9dQZF1DXa8n42306eJB",
    "rap": "3fxpDkHyW6Y2aR5FFRLOtO",
    "classical": "5tXCRZAUKp2uqtmJZNkQxY"}

X_list = [] # it will contain all the matrices X of each playlist
extra_info = {"id": [],
              "song_artist": [],
              "song_name": [],
              "original_playlist": []}
for pl_name, pl_id in playlist_ids.items():
    print(pl_name + ":\t " + pl_id, end="")
    
    #check whether or not we already have the playlist information
    filename = "downloaded_playlists/pl_info_" + pl_name + ".pickle"
    if not os.path.isfile(filename):

        # 2) Download the playlist information for each one of the above playlists.
        playlist_information = get_spotify_playlist(pl_id)

        # 3) Save to files the objects with the playlist information.
        pickle.dump(playlist_information, open("downloaded_playlists/pl_info_" + pl_name + ".pickle", "wb"))

    else:
        playlist_information = pickle.load(open(filename, "rb"))

    # 4) Create a function to extract a list of song IDs from the playlist information (only the IDs!)
    song_list = extract_song_ids(playlist_information)

    #check whether or not we already have the playlist information
    filename = "downloaded_playlists/pl_feat_" + pl_name + ".pickle"
    if not os.path.isfile(filename):
        # 5) Iterate the song IDs and download the spotify features. Store all the information.
        song_feature_list = get_feature_list(song_list)
        pickle.dump(song_feature_list, open(filename, "wb"))
    else:
        song_feature_list = pickle.load(open(filename, "rb"))

    # 6) Create data matrix X
    X_current_playlist, feature_names = create_data_matrix(song_feature_list)
    X_list.append(X_current_playlist)
    extra_info["id"] += song_list
    extra_info["song_name"] += extract_song_names(playlist_information)
    extra_info["song_artist"] += extract_song_artists(playlist_information)
    extra_info["original_playlist"] += [pl_name]*len(song_list)
    
    print("with " + str(len(song_list)) + " songs")

# 7) Concatenate all matrices X
X = np.concatenate(X_list, axis=0)
print(X.shape)

# 8) Improve data representation
# X, y = improve_data_representation(X, extra_info)

# -----------------------------------------------------
# Create 2D representation
# -----------------------------------------------------
from sklearn.manifold import Isomap, TSNE
man = TSNE()
z = man.fit_transform(X)
print(z.shape)

# -----------------------------------------------------
# Visualize it
# -----------------------------------------------------
visualize_representations(z, extra_info, with_click=True)
plt.show()


# -----------------------------------------------------
# Explore it to create the Playlist
# -----------------------------------------------------
# ref_song = "..."
# num_songs = 10
# playlist = create_playlist(ref_song, num_songs)

# -----------------------------------------------------
# Upload playlist
# -----------------------------------------------------
# upload_playlist(playlist)
