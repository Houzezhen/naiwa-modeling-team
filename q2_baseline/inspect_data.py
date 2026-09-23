import argparse
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    with open(args.path, "rb") as handle:
        data = pickle.load(handle)
    for split_name, split in data.items():
        print(split_name, "fields:", list(split), flush=True)
        for key, value in split.items():
            if hasattr(value, "shape"):
                array = np.asarray(value)
                print(key, array.shape, array.dtype, "first:", array.reshape(-1)[:5], flush=True)
            elif isinstance(value, (list, tuple)):
                print(key, "length:", len(value), "first:", str(value[:2])[:150], flush=True)
            else:
                print(key, type(value).__name__, str(value)[:120], flush=True)


if __name__ == "__main__":
    main()
