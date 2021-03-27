# List of pre-trained GANsformer networks
import pickle
import dnnlib
import dnnlib.tflib as tflib

gdrive_urls = {
    "gdrive:clevr-snapshot.pkl": "https://drive.google.com/uc?id=1JNb9r2j0MKpygxn-FT-HGClfzZ0CV_dy",
    "gdrive:cityscapes-snapshot.pkl": "https://drive.google.com/uc?id=1SLImpJEw--fKe9ngM99-KKXOXmmoWw6g",
    "gdrive:ffhq-snapshot.pkl": "https://drive.google.com/uc?id=13rF5RXN4-FeEZX9Ph40jXnt7Bqw0PJDk",
    "gdrive:bedrooms-snapshot.pkl": "https://drive.google.com/uc?id=1-2L3iCBpP_cf6T2onf3zEQJFAAzxsQne"
}

def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)

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
