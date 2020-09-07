from sjf_data_viz.spotify import *

sp = login_to_spotify()
pl1 = query_spotify_for_playlist("37i9dQZF1DWVPKP49DU8tu")

# Collect some data
music = dowload_music()

# # Create 2D representation
# X = music
# z = tsne.fit_transform(X)
#
# # Visualize it
# plt.plot(z)
#
# # Explore it to create the Playlist
# ref_song = "..."
# num_songs = 10
# playlist = create_playlist(ref_song, num_songs)
#
# # Upload playlist
# upload_playlist(playlist)