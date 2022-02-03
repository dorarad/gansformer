# List of pre-trained GANsformer networks
import pickle
import dnnlib
import dnnlib.tflib as tflib

gdrive_urls = {
    "gdrive:clevr-snapshot.pkl":        "https://drive.google.com/uc?id=1eRgwoasbDgUAA2tsD-LKghDiYTsfHWn3",
    "gdrive:cityscapes-snapshot.pkl":   "https://drive.google.com/uc?id=1Lrq3ga9N9ViH2KyvpCnSdG2xyZe_yDUA",
    "gdrive:ffhq-snapshot.pkl":         "https://drive.google.com/uc?id=1QvGFQfvPXsqsiQE5jWgRM9awxfaWnoqd",
    "gdrive:bedrooms-snapshot.pkl":     "https://drive.google.com/uc?id=1GkmnFqwUI0X5dOnSHOFDeWg_jJAc08Za"
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
