from pathlib import Path
import argparse

def update_xml(input_xml, export_dir, new_cli_path):
    # Open original file
    data = open(input_xml, 'r').read()
    x = data.replace("./data/climate/new_CAMP_MABRY_TX-hour.cli", new_cli_path)
    new_cli_path = Path(new_cli_path)
    with open(Path(export_dir) / f'{new_cli_path.stem}.xml', "w") as file:
        file.write(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modify xml file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', "--input_xml", default='./data/SRLOD3.1_Annual_results.xml',
                        type=str, help="input xml file template", required=True)
    parser.add_argument('-e', "--export_dir", default='./new_xml/',
                        type=str, help="output xml file directory", required=True)
    parser.add_argument('-c', "--climate_file", default='./test.cli',
                        type=str, help="new climate file", required=True)
    args = parser.parse_args()
    config = vars(args)
    update_xml(config['input_xml'], config['export_dir'], config['climate_file'])