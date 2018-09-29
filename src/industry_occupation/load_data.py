import pandas as pd
import numpy as np
import json


industry_occupation_file = 'industry_occupation.csv'


data_headers = ['geo_id', 'fips', 'county', 'total_employed', 'agriculture', 'construction', 'manufacturing', 'wholesale_trade', 'retail_trade', 'transport_utilities', 'information', 'finance_insurance_realestate', 'prof_scientific_waste', 'edu_health', 'arts_recreation', 'other', 'public_admin', 'year']
result_headers = ['year', 'total_employed', 'agriculture', 'construction', 'manufacturing', 'wholesale_trade', 'retail_trade', 'transport_utilities', 'information', 'finance_insurance_realestate', 'prof_scientific_waste', 'edu_health', 'arts_recreation', 'other', 'public_admin']
df_withNA =  pd.read_csv(industry_occupation_file, names=data_headers, encoding='latin-1')

df = df_withNA.dropna()

countys = df['county']
geo_ids = df['geo_id']
fips = df['fips']
years = df['year']

fips_id = fips.drop_duplicates()

print(fips)
print(fips_id)


def industry_occupation_fips_year(fips, year):
    selected = df.loc[(df['fips'] == fips) & (df['year'] == year)]
    return selected[result_headers]

industry_occupation_fips_year("1003", "2016")

def industry_occupation_fips(fips):
    selected = df.loc[(df['fips'] == fips)]
    return selected[result_headers]


data = {}

for fip in fips_id:
    data[fip] = {}
    industry_occupation = industry_occupation_fips(fip)
    for index, row in industry_occupation.iterrows():
        year = row['year']
        data[fip][year] = []
        row = row[1:]
        for item in row:
            data[fip][year].append(item)

print(data)
with open('industry_occupation.json', 'w') as outfile:
    json.dump(data, outfile)

