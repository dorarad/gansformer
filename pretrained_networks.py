# List of pre-trained GANsformer networks
import pickle
import dnnlib
import dnnlib.tflib as tflib

gdrive_urls = {
    "gdrive:clevr-snapshot.pkl":        "https://drive.google.com/uc?id=1zBh-U2kyVgN3C_P_7GqsMEHvBdz2lobu",
    "gdrive:cityscapes-snapshot.pkl":   "https://drive.google.com/uc?id=1XPGYzUP_1ETFtz5bUhpUFPha1IBNTuZh",
    "gdrive:ffhq-snapshot.pkl":         "https://drive.google.com/uc?id=1tgs-hHaziWrh0piuX3sEd8PwE9gFwlNh",
    "gdrive:bedrooms-snapshot.pkl":     "https://drive.google.com/uc?id=1BpICHESy7O0gjXK0KNES5OgZ130QzMup"
}

eval_gdrive_urls = gdrive_urls.copy()
eval_gdrive_urls.update({
    "gdrive:cityscapes-snapshot-2048.pkl":   "https://drive.google.com/uc?id=1Zw1cFxxN6-iC_M4x6Zbf9lwH9wKryW3p",
    "gdrive:ffhq-snapshot-1024.pkl":         "https://drive.google.com/uc?id=10V4yK_rQWb4F6Q4vwqkO5XNKX721k3zl"
})

def get_path_or_url(path_or_gdrive_path, eval = False):
    nets = eval_gdrive_urls if eval else gdrive_urls
    return nets.get(path_or_gdrive_path, path_or_gdrive_path)


def get_path_or_url(path_or_gdrive_path):
    return nets.get(path_or_gdrive_path, path_or_gdrive_path)

_cached_networks = dict()

def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if dnnlib.util.is_url(path_or_url):
        stream = dnnlib.util.open_url(path_or_url, cache_dir = ".GANsformer-cache")
    else:
        stream = open(path_or_url, "rb")

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding = "latin1")[:3]
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs
