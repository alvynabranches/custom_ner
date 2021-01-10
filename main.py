import numpy as np, pandas as pd
from os.path import dirname, abspath
from time import perf_counter
from pandas import read_excel
current_location = dirname(abspath(__file__))

s = perf_counter()
# df = read_excel('./data/indeed_results.xlsx')

content = 'Acknowledgment of PAN application Aadhaar Card or voter ID Benefits: Flexible working hours for security guards, computer'
    
print(f'{content[0:33]=}')

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')