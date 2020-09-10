import os
import pickle
from datetime import datetime
from sjf_data_viz.spotify import *

starting_song = 123
recompute_representation = False

# -----------------------------------------------------
# Collect some data from spotify and save them
# -----------------------------------------------------
# 1) Create a dictionary of playlist IDs like the following
playlist_ids = {
    "punk1": "37i9dQZF1DX3LDIBRoaCDQ",
    "punk2": "37i9dQZF1DXa9wYJr1oMFq",
    "reggae": "37i9dQZF1DXa8n42306eJB",
    "rap": "3fxpDkHyW6Y2aR5FFRLOtO",
    "classical": "5tXCRZAUKp2uqtmJZNkQxY",
    "classic": "37i9dQZF1DX9OZisIoJQhG",
    "rock-classic": "37i9dQZF1DWXRqgorJj26U",
    "heavy metal": "37i9dQZF1DX9qNs32fujYe",
    "jazz": "37i9dQZF1DX4wta20PHgwo",
    "funk": "4xFSdiuP4gpR4wq2OghlOs",
    "hip hop": "37i9dQZF1DX0XUsuxWHRQd",
    "electronic": "37i9dQZF1DWSFNWN7fsnAm",
    "indie": "37i9dQZF1DX8hcTuUCeYxa"}

X_list = [] # it will contain all the matrices X of each playlist
extra_info = {"id": [],
              "song_artist": [],
              "song_name": [],
              "original_playlist": []}
for pl_name, pl_id in playlist_ids.items():
    print(pl_id + ":\t" + pl_name + "\t\t", end="")
    
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
playlist_relevance = .4
X, feature_names, _ = improve_data_representation(X, feature_names, extra_info, scale=True,
                                                  use_playlist=playlist_relevance, aug_ica=False, aug_tsne=False, aug_umap=False)

# -----------------------------------------------------
# Create 2D representation
# -----------------------------------------------------
from sklearn.manifold import Isomap, TSNE, MDS
from umap import UMAP
man = TSNE(perplexity=50)
filename = "downloaded_playlists/z_" + str(man) + str(playlist_relevance) + ".pickle"
if recompute_representation or not os.path.isfile(filename):
    z = man.fit_transform(X)
    pickle.dump(z, open(filename, "wb"))
else:
    z = pickle.load(open(filename, "rb"))
print(z.shape)

# -----------------------------------------------------
# Visualize its
# -----------------------------------------------------
# visualize_representations(z, extra_info, with_click=True)
# plt.title(man)
# plt.show()


# -----------------------------------------------------
# Explore it to create the Playlist
# -----------------------------------------------------
s0 = starting_song
playlist = create_playlist(z, s0)

for song_index in playlist:
    print(extra_info["id"][song_index])

for song_index in playlist:
    print(extra_info["song_name"][song_index] + "  |  " + extra_info["song_artist"][song_index])

song_ids = []
for song_index in playlist:
    song_ids.append(extra_info["id"][song_index])

pickle.dump(song_ids, open("created_playlists/" + str(datetime.now()) + ".pickle", "wb"))

# -----------------------------------------------------
# Visualize playlist
# -----------------------------------------------------
visualize_representations(z, extra_info, with_click=True)
plt.scatter(z[s0, 0], z[s0, 1], c="g", marker="+", label="start", s=200)
plt.plot(z[playlist, 0], z[playlist, 1], c="r", marker="", label="playlist")

# plt.plot(z[song_ids, ])
plt.title(man)
plt.savefig("playlist.pdf")
plt.show()

# -----------------------------------------------------
# Upload playlist
# -----------------------------------------------------
res = input("Do you want to upload it? yes/[no]\n")
if res == "yes":
    upload_playlist(song_ids)
