import pickle

def read_pkl(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)