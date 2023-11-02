from utils.misc import *
from utils.vis import plot_original_load_data, plot_original_cli_data

# res_path = 'data/citydnn/Portland_OR-hour/Portland_OR-hour_TH.out'
# res_df = read_result_file(res_path)
# import pdb; pdb.set_trace()
# plot_original_load_data(res_df)

cli_path = 'new_cli/citydnn/Portland_OR-hour.cli'
cli_df = read_climate_file(cli_path)
plot_original_cli_data(cli_df)
