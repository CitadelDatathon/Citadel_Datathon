import pandas as pd
import numpy as np
import json

earning_file = 'earnings.csv'


data_headers = ['geo_id','fips','county','total_med','total_agri_fish_mine','agri_fish_hunt','mining_quarrying_oilgas_extract','construction','manufacturing','wholesale_trade','retail_trade','transport_warehouse_utilities','transport_warehouse','utilities','information','fin_ins_realest','fin_ins','realest_rent_lease','total_prof_sci_mgmt_admin','prof_sci_tech','mgmt','admin_sup','total_edu_health_social','edu_serv','health_social','total_arts_ent_acc_food','arts_ent_rec','acc_food_serv','other_ser','pub_admin','year']

result_headers = ['year', 'total_med','total_agri_fish_mine','agri_fish_hunt','mining_quarrying_oilgas_extract','construction','manufacturing','wholesale_trade','retail_trade','transport_warehouse_utilities','transport_warehouse','utilities','information','fin_ins_realest','fin_ins','realest_rent_lease','total_prof_sci_mgmt_admin','prof_sci_tech','mgmt','admin_sup','total_edu_health_social','edu_serv','health_social','total_arts_ent_acc_food','arts_ent_rec','acc_food_serv','other_ser','pub_admin']
df_withNA =  pd.read_csv(earning_file, encoding='latin-1')

df = df_withNA.dropna()


countys = df['county']
fips = df['fips']
fips_id = fips.drop_duplicates()
years = df['year']



def earning_fips_year(fips):
    selected = df.loc[(df['fips'] == fips)]
    if selected.empty:
        return np.zeros(result_headers.size)
    return selected[result_headers]



data = {}

for fip in fips_id:
    data[fip] = {}
    industry_occupation = earning_fips_year(fip)
    for index, row in industry_occupation.iterrows():
        year = row['year']
        data[fip][year] = []
        row = row[1:]
        for item in row:
            data[fip][year].append(item)

print(data)
with open('earnings.json', 'w') as outfile:
    json.dump(data, outfile)

