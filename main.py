import numpy as np, pandas as pd
from os.path import dirname, abspath
from time import perf_counter
from pandas import read_excel
current_location = dirname(abspath(__file__))

s = perf_counter()
# df = read_excel('./data/indeed_results.xlsx')

content = 'Documents to be carried for Walk-in: Valid Two-wheeler Driver License and Driver RC book PAN Card or'
    
print(f'{content[74:88]=}')

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')