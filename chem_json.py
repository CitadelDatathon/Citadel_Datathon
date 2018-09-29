import os
import pandas as pd
import json

PATH = os.getcwd()
DATA_DIR = os.path.join(PATH, 'Datathon Materials 2')
DROUGHT_DIR = os.path.join(DATA_DIR, 'droughts.csv')
EARNING_DIR = os.path.join(DATA_DIR, 'earnings.csv')
EDU_DIR = os.path.join(DATA_DIR, 'education_attainment.csv')
INDUSTRY_DIR = os.path.join(DATA_DIR, 'industry_occupation.csv')
CHEM_DIR = os.path.join(DATA_DIR, 'chemicals.csv')

df_drought = pd.read_csv(DROUGHT_DIR)
df_chem = pd.read_csv(CHEM_DIR)

# print(df_chem[['fips', 'chemical_species', 'value', 'year']])

df_drought['dsci'] = df_drought['d0'] + 2* df_drought['d1'] + 3* df_drought['d2']  + 4* df_drought['d3'] + 5* df_drought['d4']
# print(df_drought[['fips', 'dsci', 'valid_start']])


data = {}
for index, row in df_chem.iterrows():
    fips = row['fips']
    year = row['year']
    chem = row['chemical_species']
    value = row['value']

    if fips not in data:
        data[fips] = {}

    if year not in data[fips]:
        data[fips][year] = {}

    data[fips][year][chem] = value

with open('chem.json', 'w') as outfile:
    json.dump(data, outfile)




