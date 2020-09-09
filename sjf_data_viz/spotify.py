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
    assert len(feat_) == 0
    return feat_[0]

def extract_song_ids(pl_info):
    playlist_song_ids = []
    for i in range(50):
        playlist_song_ids.append(pl_info["tracks"]["items"][i]["track"]["id"])
    return playlist_song_ids

def get_feature_list(song_list):
    song_feature_list = []
    for song_id in song_list:
        song_features = get_spotify_features(song_id)
        song_feature_list.append(song_features)
    return song_feature_list

