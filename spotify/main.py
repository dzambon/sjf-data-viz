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

for pl_name, pl_id in playlist_ids.items():
    print(pl_name + "-->" + pl_id)
    
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
   

# -----------------------------------------------------
# Create 2D representation
# -----------------------------------------------------
# X = music
# z = tsne.fit_transform(X)

# -----------------------------------------------------
# Visualize it
# -----------------------------------------------------
# plt.plot(z)

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