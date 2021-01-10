import pandas as pd
from os.path import dirname, abspath
from time import perf_counter
from pandas import read_excel
current_location = dirname(abspath(__file__))

s = perf_counter()
# df = read_excel('./data/indeed_results.xlsx')

content = 'account executives night shifts weekend Attractive Weekly, weekend and monthly Incentives for 10th pass, 12th pass'
    
print(f'{content[94:103]=}')

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')