import pandas as pd
import numpy as np
import json

education_attainment_file = 'education_attainment.csv'



result_headers = ['year', 'less_than_hs', 'hs_diploma', 'some_college_or_associates', 'college_bachelors_or_higher', 'pct_less_than_hs', 'pct_hs_diploma', 'pct_college_or_associates', 'pct_college_bachelors_or_higher']
df =  pd.read_csv(education_attainment_file, encoding='latin-1')

print(df)


def education_attainment_fips_year(fips):
    selected = df.loc[(df['fips'] == fips)]
    print(selected)
    return selected




