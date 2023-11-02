from utils.misc import *
from pathlib import Path

def shorten_result_file(file_path, random_arg, cli_arg):
    df = read_result_file(file_path)
    n_export_dir = Path(f'/work/08388/tudai/ls6/citySimOutput/export_csv/{random_arg}')
    n_export_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(n_export_dir / f'{cli_arg}.csv')

if __name__ == '__main__':
    export_dir = Path('/work/08388/tudai/ls6/citySimOutput/export/')
    file_list = export_dir.glob('random_*/**/*_TH.out')
    file_list = [x for x in file_list if x.is_file()]
    s_file = file_list[0]
    for f in file_list:
        parts = f.parts
        shorten_result_file(f, parts[-3], parts[-2])
