import os
import pandas as pd
import json
from datetime import datetime as dt

PATH = os.getcwd()
DATA_DIR = os.path.join(PATH, 'Datathon Materials 2')
DROUGHT_DIR = os.path.join(DATA_DIR, 'droughts.csv')
EARNING_DIR = os.path.join(DATA_DIR, 'earnings.csv')
EDU_DIR = os.path.join(DATA_DIR, 'education_attainment.csv')
INDUSTRY_DIR = os.path.join(DATA_DIR, 'industry_occupation.csv')
CHEM_DIR = os.path.join(DATA_DIR, 'chemicals.csv')

df_drought = pd.read_csv(DROUGHT_DIR)


print(df_drought[['fips', 'valid_start']])

data = {}
for index, row in df_drought.iterrows():
    fips = row['fips']
    none = row['none']
    d0 = row['d0']
    d1 = row['d1']
    d2 = row['d2']
    d3 = row['d3']
    d4 = row['d4']

    valid_start = dt.strptime(row['valid_start'], '%Y-%m-%d')
    year = valid_start.year

    if fips not in data:
        data[fips] = {}

    if year not in data[fips]:
        data[fips][year] = [none, d0, d1, d2, d3, d4]
    else:
        data[fips][year][0] += none
        data[fips][year][1] += d0
        data[fips][year][2] += d1
        data[fips][year][3] += d2
        data[fips][year][4] += d3


with open('drought.json', 'w') as outfile:
    json.dump(data, outfile)


