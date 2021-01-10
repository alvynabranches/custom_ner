try:
    pass
except ModuleNotFoundError:
    pass

import en_core_web_sm
from os.path import dirname, abspath
from time import perf_counter
from pandas import read_excel
current_location = dirname(abspath(__file__))

s = perf_counter()

df = read_excel('./data/indeed_results.xlsx')

description = df['Description'].dropna().reset_index()['Description']
pre_processed=description.apply(lambda x: str(x).replace('Job Summary','').replace('Job Description','').replace('Short Description','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('/',' ').replace('|',' ').replace('  ',' ').replace('"',' ').replace("'",' ').replace('  ','').lstrip().rstrip())

nlp = en_core_web_sm.load()

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')