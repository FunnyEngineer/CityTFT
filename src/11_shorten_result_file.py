from utils.misc import *

def shorten_result_file(file_path):
    df = read_result_file(file_path)
    df.to_csv('test.csv')

if __name__ == '__main__':
    shorten_result_file('data/random_urban/random_1/Bogota_El-dorado-hour/Bogota_El-dorado-hour_TH.out')