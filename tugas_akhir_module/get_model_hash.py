import hashlib

from .read_yaml import config_model_location

def get_model_hash(name: str):
    filename = config_model_location()+name
    with open(filename,"rb") as f:
    bytes = f.read() # read entire file as bytes
    readable_hash = hashlib.sha256(bytes).hexdigest();

    return readable_hash