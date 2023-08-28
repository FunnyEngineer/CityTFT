import matplotlib.pyplot as plt
import pandas as pd
import pdb
from pathlib import Path

def visualize_result():
    result = pd.read_csv('./data/result/SRLOD3.1_Annual_results_TH.out', sep='\t')
    # result = result.dropna(axis=0, how='any')
    # result = result.reset_index(drop=True)
    result2 = pd.read_csv('./data/result_2/SRLOD3.1_Annual_results_TH.out', sep='\t')
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for ax, col in zip(axs, result.columns[2:4]):
        ax.set_ylabel(col)
        ax.plot(result.index, result[col], label='Original Climate File', color='grey')
        ax.plot(result.index, result2[col], label='New Climate File', color='teal')
        ax.legend()
        
    plt.plot(result.index[:24*60], result.iloc[:24*60, 2], label='Original Climate File', color='grey')
    plt.plot(result.index[:24*60], result2.iloc[:24*60, 2], label='New Climate File', color='teal')
    plt.legend()
    plt.show()
    pdb.set_trace()

def compare_climate_change_scenario():
    res1 = pd.read_csv('export/CAMP_MABRY_TX-RCP2.6-2030/CAMP_MABRY_TX-RCP2.6-2030_TH.out', sep='\t')
    res2 = pd.read_csv('export/CAMP_MABRY_TX-RCP4.5-2030/CAMP_MABRY_TX-RCP4.5-2030_TH.out', sep='\t')
    res3 = pd.read_csv('export/Bogota_El-dorado-hour/Bogota_El-dorado-hour_TH.out', sep='\t')

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    start = 24 * 60
    end = 24* 120
    for ax, col in zip(axs, res1.columns[2:4]):
        ax.set_ylabel(col)
        ax.plot(res1.index[start:end], res1[col][start:end], label='RCP2.6', color='grey')
        ax.plot(res3.index[start:end], res3[col][start:end], label='Bogota_El', color='red')
        ax.legend()
        
    plt.legend()
    plt.show()
    pdb.set_trace()

def compare_average_temperature():
    sce = ['RCP2.6', 'RCP4.5', 'RCP8.5']
    years = ['2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    for s in sce:
        aver_temp = []
        for y in years:
            res = pd.read_csv(f'data/climate/AUS_cli/CAMP_MABRY_TX-{s}-{y}.cli', sep='\t', skiprows=3)
            aver_temp.append(res[' Ts'].mean())
        axs.plot(years, aver_temp, label=s)
    plt.legend()
    plt.show()

def compare_different_climate_result():
    export_dir = 'export/citydnn'
    export_dir = Path(export_dir)
    # get TH.out file for each city
    result_files = list(export_dir.glob('**/*TH.out'))
    # get city name
    city_names = [p.stem.split('-')[0] for p in result_files]

    # init fig
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # plot each city
    start = 24 * 30 * 6
    end = 24 * 30 * 8
    hourly_date = pd.date_range('1/1/2011', periods=24*365, freq='H')
    for city_name, result_file in zip(city_names, result_files):
        res = pd.read_csv(result_file, sep='\t')
        col = res.columns[3]
        axs.plot(hourly_date[start:end], res[col][start:end], label=city_name)
    plt.title(col)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # compare_climate_change_scenario()
    # compare_average_temperature()
    compare_different_climate_result()