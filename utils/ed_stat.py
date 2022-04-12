#
import sys
import os
import numpy as np
import pandas as pd
import time


def random_pick_files(dir, sample_num = 1000):
    filenames = os.listdir(dir)
    max_n = len(filenames)
    sample_num = min(sample_num, max_n)
    sample_ids = np.random.randint(0,sample_num-1, size=(sample_num,))
    pick_names = np.array(filenames)[sample_ids]
    return pick_names.tolist()

def read_to_numpy(dir, pick_names):
    assert len(pick_names) > 0
    data = np.load(os.path.join(dir, pick_names[0]))
    max_row = int(1e5*len(pick_names))
    result = np.zeros(shape=(max_row, data.shape[1]))
    count = 0
    for name in pick_names:
        data = np.load(os.path.join(dir, name))
        if count+data.shape[0] > max_row: break
        result[count:count+data.shape[0],:] = data
        count += data.shape[0]
    count = min(max_row, count)
    return result[:count]

def stat(nparray):
    df = pd.DataFrame(nparray)
    print(df.describe(percentiles=[.001, .01, .05, .1, .5, .9, .99, .999]))

if __name__ == '__main__':
    dir = sys.argv[1]
    b = time.time()
    pick_names = random_pick_files(dir)
    e = time.time()
    print("random_pick_files cost: {:.3f}".format(e-b))
    res = read_to_numpy(dir, pick_names)
    e1 = time.time()
    print("read_to_numpy cost: {:.3f}".format(e1-e))
    stat(res)
    e2 = time.time()
    print("stat cost: {:.3f}".format(e2-e1))
