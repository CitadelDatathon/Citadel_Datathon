import pandas as pd
import numpy as np

water_usage_file = 'water_usage.csv'

data_headers = ['state', 'state_fips', 'county', 'county_fips', 'fips', 'year', 'population', 'ps_groundwater',	'ps_surfacewater', 'ps_total', 'd_selfsupplied', 'd_totaluse', 'ir_sprinkler', 'ir_microirrig',	'ir_surface', 'ir_total', 'crop_ir_sprinkler', 'crop_ir_microirrig', 'crop_ir_surface',	'crop_ir_total', 'therm_power',	'therm_power_oncethrough',	'therm_power_recirc']
df_withNA =  pd.read_csv(water_usage_file, names=data_headers, encoding='latin-1')

df = df_withNA.dropna()

fips = df['fips']
years = df['year']
countys = df['county']
populations = df['population']


print(fips)
print(populations)

df_group = df.groupby().get_group(0)
print(df_group)
