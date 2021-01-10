import numpy as np, pandas as pd
from os.path import dirname, abspath
from time import perf_counter
current_location = dirname(abspath(__file__))

s = perf_counter()
# df = pd.read_excel('./data/indeed_results.xlsx')

content = 'with prior experience in Data entry, Call Centre, Admin, Collection Agents, Collection Executives, Customer Support Executive, Office Assistant, Driver, Delivery Boys, Back Office, Teacher, Banking, Accounts, Operator can apply to maximize their earnings up to 25,000 per month.'
    
print(f'{content[190:197]=}')

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')