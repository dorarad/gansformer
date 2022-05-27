# List of pre-trained GANsformer networks
import pickle
import dnnlib
import dnnlib.tflib as tflib

gdrive_urls = {
    "gdrive:clevr-snapshot.pkl":        "https://drive.google.com/uc?id=1sYtuNEi0HGBH1F8GW2JlF4mCRGGrtZa9",
    "gdrive:cityscapes-snapshot.pkl":   "https://drive.google.com/uc?id=1iL8S26IYCNAQmUS-6GYCVheKFtCE4ktQ",
    "gdrive:ffhq-snapshot.pkl":         "https://drive.google.com/uc?id=1-b0vwevUQs6LI_EybdO8XJ5uYSx63vEa",
    "gdrive:bedrooms-snapshot.pkl":     "https://drive.google.com/uc?id=1lrB4t3hOOpn8YiwHEduIZYzndHrsKX8z"
}

eval_gdrive_urls = gdrive_urls.copy()
eval_gdrive_urls.update({
    "gdrive:cityscapes-snapshot-2048.pkl":   "https://drive.google.com/uc?id=1lJlbIuxF89-owG7vQSBYXc8y2RqqP_bP",
    "gdrive:ffhq-snapshot-1024.pkl":         "https://drive.google.com/uc?id=1OcsibUthk2y8y0slf2lQnrYBAtmdJYCQ"
})

def get_path_or_url(path_or_gdrive_path, eval = False):
    nets = eval_gdrive_urls if eval else gdrive_urls
    return nets.get(path_or_gdrive_path, path_or_gdrive_path)

_cached_networks = dict()

def load_networks(path_or_gdrive_path, eval = False):
    path_or_url = get_path_or_url(path_or_gdrive_path, eval)
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
