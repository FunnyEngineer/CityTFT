import pandas as pd
import yaml
from .misc import *
import numpy as np
import pdb
from configs.configuration import H_MEAN, H_STD, C_MEAN, C_STD


def combine_res_bud_cli(cli_path, res_path, bud_path):
    bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                        Loader=yaml.FullLoader)['BUD_PROPS']
    cli = read_climate_file(cli_path)
    res = normalize_load(read_result_file_csv(
        res_path), H_MEAN, H_STD, C_MEAN, C_STD)
    bud = read_building_info(bud_path)[bud_key]
    total = concat_res_bud_cli_by_order(cli, res, bud, bud_key)
    return total

def concat_res_bud_cli_by_id(cli, res, bud, bud_key):
    res_id_list = np.array([int(i.split('(')[0])
                           for i in res.columns.to_list()])
    total = pd.DataFrame(columns=cli.columns.to_list()
                         + bud_key + ['Heating', 'Cooling'])
    for ind_bud, bud_row in bud.iterrows():
        sin_bud_df = cli.copy()
        for bud_prop, bud_val in bud_row.items():
            sin_bud_df[bud_prop] = bud_val

        res_sub_df = res.loc[:, res_id_list == ind_bud]
        sin_bud_df['Heating'] = res_sub_df.iloc[:, 0]
        sin_bud_df['Cooling'] = res_sub_df.iloc[:, 1]
        total = pd.concat([total, sin_bud_df])
    total = total.reset_index(drop=True)


def concat_res_bud_cli_by_order(cli, res, bud, bud_key):
    total = pd.DataFrame(columns=cli.columns.to_list()
                         + bud_key + ['Heating', 'Cooling'])
    bi=0
    for _, bud_row in bud.iterrows():
        sin_bud_df = cli.copy()
        for bud_prop, bud_val in bud_row.items():
            sin_bud_df[bud_prop] = bud_val

        res_sub_df = res.iloc[:, bi:bi+2]
        sin_bud_df['Heating'] = res_sub_df.iloc[:, 0]
        sin_bud_df['Cooling'] = res_sub_df.iloc[:, 1]
        total = pd.concat([total, sin_bud_df])
        bi += 2
    total = total.reset_index(drop=True)

if __name__ == '__main__':
    cli_path = 'new_cli/citydnn/Bogota_El-dorado-hour.cli'
    res_path = 'data/citydnn/Bogota_El-dorado-hour/Bogota_El-dorado-hour_TH.out'
    bud_path = 'data/ut_building_info.csv'

    combine_res_bud_cli(cli_path, res_path, bud_path)
