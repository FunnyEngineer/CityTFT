import csv
import argparse
import pandas as pd
from pathlib import Path
import pdb

def convert_geolocation_to_csv(cli_dir = 'new_cli/citydnn', export_file_path = 'data/cli_loc.csv'):
    cli_dir = Path(cli_dir)
    cli_files = list(cli_dir.glob('*.cli'))
    cli_files.sort()
    # pd.DataFrame(columns=['lon', 'lat', 'elevation', 'timezone'])
    data = {}
    file_dir = []
    for cli_file in cli_files:
        loc = pd.read_csv(cli_file, nrows=1, index_col=False, header=None).values.item()
        info = pd.read_csv(cli_file, skiprows=1, nrows=1, index_col=False, header=None).values
        # print(info.shape)
        data[loc] = info.flatten()
        file_dir.append(cli_file.stem)
    data = pd.DataFrame.from_dict(data, orient='index', columns=['lat', 'lon', 'elevation', 'timezone'])

    cz_df = pd.read_csv('data/cli_climate_zone.csv')
    data = data.merge(cz_df, left_index=True, right_on='Location')
    data = data[['Location', 'lat', 'lon', 'elevation', 'timezone', 'Climate Abbreviation', 'Climate Full Name']]
    data['file_dir'] = file_dir
    data = data.rename(columns={'Climate Abbreviation': 'Short Climate Zone'})
    data = data.rename(str.lower, axis='columns')
    data['Koppen'] = data['short climate zone'].str[0]

    data.to_csv(export_file_path)
    pdb.set_trace()

if __name__ == '__main__':
    convert_geolocation_to_csv()