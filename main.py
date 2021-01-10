import numpy as np, pandas as pd
from os.path import dirname, abspath
from time import perf_counter
current_location = dirname(abspath(__file__))

s = perf_counter()



e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')