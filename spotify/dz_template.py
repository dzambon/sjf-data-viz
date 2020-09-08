from sjf_data_viz.spotify import *

sp = login_to_spotify()

# -----------------------------------------------------
# Collect some data from spotify and save them
# -----------------------------------------------------
# 1) Create a dictionary of playlist IDs like the following
#         >>> playlist_ids = {
#         >>>     "punk": "37i9dQZF1DX3LDIBRoaCDQ",
#         >>>     "reggae": "37i9dQZF1DXa8n42306eJB",
#         >>> }

# 2) Download the playlist information for each one of the above playlists.
# hint:
#   - Make a for loop and iterate through the playlist ids. you can follow this
#     example https://www.tutorialspoint.com/How-to-iterate-through-a-dictionary-in-Python
#   - Use the following function to download the information
#         >>> playlist_information = get_spotify_playlist("37i9dQZF1DX3LDIBRoaCDQ")

# 3) Save to files the objects with the playlist information.
# hints:
#   - create a folder `downloaded_playlists`
#   - save each playlist info on a different file, so do another for loop.
#   - use `pickle` package to save the playlist information to file. A simple example
#     https://wiki.python.org/moin/UsingPickle.
#     For example, the punk playlist can be saved as follows:
#         >>> pickle.dump(playlist_information, open("downloaded_playlists/playlist_info_{}.p".format("punk"), "wb" ))

# 4) Create a function to extract a list of song IDs from the playlist information (only the IDs!)
# hints:
#   - playlist_information is a dictionary. Explore it and find the list of IDs
#   - store them in list:
#         >>> playlist_songs = []
#         >>> for song_id in playlist_information.........:
#         >>>     playlist_songs.append(song_id)

# 5) Iterate the song IDs and download the spotify features. Store all the information.
# hints:
#         >>> song_feature = get_spotify_features(song_id)

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