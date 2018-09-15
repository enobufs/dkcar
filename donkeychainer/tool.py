import os
import time

def make_output_dir(path):
    out_dir = './{}/{}'.format(path, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir

