import os
import pickle


def bindump(data, filepath: str, overwrite=True):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    if os.path.exists(filepath) and not overwrite:
        return
    binfile = open(filepath, 'wb')
    pickle.dump(data, binfile)
    binfile.close()
