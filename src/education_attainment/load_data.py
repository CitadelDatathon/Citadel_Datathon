import pandas as pd
import numpy as np

education_attainment_file = 'education_attainment.csv'


data_headers = ['fips','state','county', 'year', 'less_than_hs', 'hs_diploma', 'some_college_or_associates', 'college_bachelors_or_higher', 'pct_less_than_hs', 'pct_hs_diploma	pct_college_or_associates', 'pct_college_bachelors_or_higher']

result_headers = ['less_than_hs','hs_diploma','some_college_or_associates','college_bachelors_or_higher','pct_less_than_hs','pct_hs_diploma','pct_college_or_associates','pct_college_bachelors_or_higher']
df_withNA =  pd.read_csv(education_attainment_file, encoding='latin-1')

df = df_withNA.dropna()
print(df)

countys = df['county']
fips = df['fips']
years = df['year']



def education_attainment_fips_year(fips, year):
    if(int(year) >= 2012 and int(year) <= 2016):
        print("2012-2016")
        year = "2012-2016"
    selected = df.loc[(df['fips'] == fips) & (df['year'] == year)]
    print(df)
    print(selected)
    return selected

