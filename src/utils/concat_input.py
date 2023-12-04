import pandas as pd
import yaml
from misc import *
import numpy as np
import pdb


def combine_res_bud_cli(cli_path, res_path, bud_path):
    bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                        Loader=yaml.FullLoader)['BUD_PROPS']
    cli = read_climate_file(cli_path)
    res = read_result_file(res_path)
    res_id_list = np.array([int(i.split('(')[0])
                           for i in res.columns.to_list()])
    bud = read_building_info(bud_path)[bud_key]
    total = pd.DataFrame(columns=cli.columns.to_list()
                         + bud_key + ['Heating', 'Cooling'] )
    for ind_bud, bud_row in bud.iterrows():
        sin_bud_df = cli.copy()
        for bud_prop, bud_val in bud_row.items():
            sin_bud_df[bud_prop] = bud_val

        res_sub_df = res.loc[:, res_id_list == ind_bud]
        sin_bud_df['Heating'] = res_sub_df.iloc[:, 0]
        sin_bud_df['Cooling'] = res_sub_df.iloc[:, 1]
        total = pd.concat([total, sin_bud_df])
    total = total.reset_index(drop=True)
    return total


if __name__ == '__main__':
    cli_path = 'new_cli/citydnn/Bogota_El-dorado-hour.cli'
    res_path = 'data/citydnn/Bogota_El-dorado-hour/Bogota_El-dorado-hour_TH.out'
    bud_path = 'data/ut_building_info.csv'

    combine_res_bud_cli(cli_path, res_path, bud_path)
