import csv
import argparse


def strip_starting_space(input_file, output_file):
    open(output_file, 'w').close() # clear the file
    with open(output_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n\n')
        with open(input_file) as file:
            tsv_file = csv.reader(file, delimiter="\t")

            # printing data line by line
            for i, line in enumerate(tsv_file):
                if i == 1:
                    writer.writerow([','.join(line[0].split())])
                else:
                    writer.writerow([v.strip()for v in line])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processing climate file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', "--input-file", default='./test_input.cli',
                        type=str, help="input climate file", required=True)
    parser.add_argument('-o', "--output-file", default='./test.cli',
                        type=str, help="output climate file", required=True)
    args = parser.parse_args()
    config = vars(args)
    strip_starting_space(config['input_file'], config['output_file'])
