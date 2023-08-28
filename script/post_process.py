import os
from os.path import join, exists
import argparse


def move_result(original_dir, new_dir, climate_name):
    # make a folder to store the results
    if not exists(join(new_dir, climate_name)):
        os.makedirs(join(new_dir, climate_name))
    # move the results to the new folder
    for file in os.listdir(original_dir):
        if file.endswith(("TH.out", "YearlyResults.out", "YearlyResultsPerBuilding.out")) and file.startswith(climate_name):
            os.rename(join(original_dir, file), join(new_dir, climate_name, file))
        elif file.endswith(".xml"):
            continue
        else:
            os.remove(join(original_dir, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post process the result",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ori", help="Source location")
    parser.add_argument("new", help="Destination location")
    parser.add_argument("cli", help="Destination location")
    args = parser.parse_args()
    config = vars(args)
    move_result(config['ori'], config['new'], config['cli'])
