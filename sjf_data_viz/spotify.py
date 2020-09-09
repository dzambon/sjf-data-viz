import numpy as np
from matplotlib import pyplot as plt
import spotipy
from .config import USERNAME, CID, SECRET
SPOTIFY_CONNECTION = None

def login_to_spotify():
    global SPOTIFY_CONNECTION, USERNAME, CID, SECRET

    if SPOTIFY_CONNECTION is not None:
        return SPOTIFY_CONNECTION

    #for avaliable scopes see https://developer.spotify.com/web-api/using-scopes/
    scope = 'user-library-read playlist-modify-public playlist-read-private playlist-modify-private'
    redirect_uri='https://developer.spotify.com/dashboard/applications/a8b92e6f19944ccb8380423f3198d1dd' # Paste your Redirect URI here

    client_credentials_manager = spotipy.oauth2.SpotifyClientCredentials(client_id=CID, client_secret=SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    token = spotipy.util.prompt_for_user_token(USERNAME, scope, CID, SECRET, redirect_uri)

    if token:
        print("Got token for", USERNAME)
        SPOTIFY_CONNECTION = spotipy.Spotify(auth=token)
        return SPOTIFY_CONNECTION
    else:
        print("Can't get token for", USERNAME)
        return None

def get_spotify_playlist(pl_id):
    global SPOTIFY_CONNECTION, USERNAME
    
    if SPOTIFY_CONNECTION is None:
        SPOTIFY_CONNECTION = login_to_spotify()
    
    pl_ = SPOTIFY_CONNECTION.user_playlist(USERNAME, pl_id)
    
    return pl_

def get_spotify_features(song_id):
    global SPOTIFY_CONNECTION
    if SPOTIFY_CONNECTION is None:
        SPOTIFY_CONNECTION = login_to_spotify()

    feat_ = SPOTIFY_CONNECTION.audio_features(song_id)
    assert len(feat_) == 1
    return feat_[0]


# def improve_data_representation(music_df, scale=False,
#                                 use_playlist=False, aug_ica=False, aug_tsne=False, aug_umap=False):
def improve_data_representation(X, feature_names, extra_info, scale=False,
                                use_playlist=False, aug_ica=False, aug_tsne=False, aug_umap=False):
    y, playlist_names = pd.factorize(extra_info["original_playlist"])
    
    cols = feature_names
    
    if use_playlist:
        oh = np.zeros((y.shape[0], y.max() + 1))
        oh[np.arange(y.shape[0]), y] = 1
        X = np.concatenate([X, oh], axis=1)
        cols += list(playlist_names)
    
    if aug_ica:
        from sklearn.decomposition import FastICA
        n_components = 3
        ica = FastICA(n_components=n_components)
        x_ = ica.fit(X).transform(X)
        X = np.concatenate([X, x_], axis=1)
        cols += ["ica{}".format(i + 1) for i in range(n_components)]
    
    if aug_tsne:
        from sklearn.manifold import TSNE
        n_components = 3
        tsne = TSNE(n_components=n_components)
        x_ = tsne.fit_transform(X)
        X = np.concatenate([X, x_], axis=1)
        cols += ["tsne{}".format(i + 1) for i in range(n_components)]
    
    if aug_umap:
        n_components = 3
        n_neighbors = 10
        from umap import UMAP
        umap = UMAP(n_components=n_components, n_neighbors=n_neighbors)
        x_ = umap.fit_transform(X)
        X = np.concatenate([X, x_], axis=1)
        cols += ["umap{}".format(i + 1) for i in range(n_components)]
    
    if scale:
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    
    return X, y, cols, scaler

def visualize_representations(z, y, playlist_ids, music_df, with_click=False):

    fig, ax = plt.subplots()
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, picker=5)
    legend1 = ax.legend(scatter.legend_elements()[0], playlist_ids,
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    ax.axis("equal")
    plt.legend()
    print(scatter.legend_elements())

    if with_click:
        current_label = ax.annotate("None", xy=z[0])
    
        def onpick(event):
            for i in event.ind:
                print("[{}] {} | {}\nhttps://open.spotify.com/track/{}".format(i,
                    music_df.track_artist[i],
                    music_df.track_name[i],
                    music_df.index[i]))
            current_label.set_text("{}_{}".format(str(i), music_df.track_name[i]))
            current_label.set_x(z[i, 0])
            current_label.set_y(z[i, 1])
            current_label.figure.canvas.draw()
    
        fig.canvas.mpl_connect('pick_event', onpick)

def extract_song_ids(pl_info):
    # Michelle and Samuel
    playlist_song_ids = []
    for i in range(len(pl_info["tracks"]["items"])):
        playlist_song_ids.append(pl_info["tracks"]["items"][i]["track"]["id"])
    return playlist_song_ids


def extract_song_artists(pl_info):
    # Michelle
    playlist_song_artists = []
    for i in range(len(pl_info["tracks"]["items"])):
        playlist_song_artists.append(pl_info["tracks"]["items"][i]["track"]["artists"])
    return playlist_song_artists

def extract_song_names(pl_info):
    # Samuel
    playlist_song_names = []
    for i in range(len(pl_info["tracks"]["items"])):
        playlist_song_names.append(pl_info["tracks"]["items"][i]["track"]["name"])
    return playlist_song_names

def get_feature_list(song_list):
    # Michelle
    song_feature_list = []
    for song_id in song_list:
        song_features = get_spotify_features(song_id)
        song_feature_list.append(song_features)
    return song_feature_list

def create_data_matrix(feature_list):
    # Samuel
    desired_features = ['danceability', 'energy', 'key', 'loudness',
                        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo']

    X = np.zeros((len(feature_list), len(desired_features)))
    for s in range(len(feature_list)):
        for f in range(len(desired_features)):
            X[s, f] = feature_list[s][desired_features[f]]
    return X, desired_features

