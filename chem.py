import json

with open('chem.json') as f:
    chem_data = json.load(f)


chemial = {}
curr_max = 0
for fip in chem_data:
    for year in chem_data[fip]:
        chem_dict = chem_data[fip][year]
        for chem in chem_dict:
            if chem not in chemial:
                chemial[chem] = curr_max
                curr_max += 1

data = {}
for fip in chem_data:
    data[fip] = {}
    for year in chem_data[fip]:
        chem_dict = chem_data[fip][year]
        row = [0] * curr_max
        for chem in chem_dict:
            row[chemial[chem]] = chem_dict[chem]
        data[fip][year] = row

print(chemial)
with open('chem_2.json', 'w') as outfile:
    json.dump(data, outfile)


