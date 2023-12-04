from utils.misc import *
from pathlib import Path
import matplotlib.pyplot as plt
import pdb

def shorten_result_file(file_path, random_arg, cli_arg):
    df = read_result_file(file_path)
    n_export_dir = Path(f'/work/08388/tudai/ls6/citySimOutput/export_csv/{random_arg}')
    n_export_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(n_export_dir / f'{cli_arg}.csv')

def visual_bud_res():
    bud = pd.read_csv('data/random_urban/random_1.csv')
    res = pd.read_csv('data/random_urban/export_csv/random_1/Bogota_El-dorado-hour.csv', index_col=0)

    b_index = bud[bud.Roof_z == 3.5].index
    fig, ax = plt.subplots(5, 1, figsize=(20, 10))
    end_point = 24 * 7
    for i in range(5):
        ax[i].bar(res.index[:end_point], res.iloc[:end_point, b_index[i]*2], color='r')
        ax[i].bar(res.index[:end_point], res.iloc[:end_point, b_index[i]*2+1], color='b')
    plt.show()
    pdb.set_trace()

if __name__ == '__main__':
    # export_dir = Path('/work/08388/tudai/ls6/citySimOutput/export/')
    # file_list = export_dir.glob('random_*/**/*_TH.out')
    # file_list = [x for x in file_list if x.is_file()]
    # s_file = file_list[0]
    # for f in file_list:
    #     parts = f.parts
    #     shorten_result_file(f, parts[-3], parts[-2])
    visual_bud_res()