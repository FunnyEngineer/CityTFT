import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pdb
from pickle import load

def read_climate_file(file_path):
    cli = pd.read_csv(file_path, sep='\t', skiprows=6)
    cli, _ = scaling_df(cli, type='cli')
    return cli

def read_result_file(file_path):
    res = pd.read_csv(file_path, sep='\t') # .dropna(axis=1, how='any')
    res = res.loc[:, (res.columns.str.contains('Heating')| res.columns.str.contains('Cooling'))]
    res = res.fillna(0) # since there is only 0.04% nan values in total, should be fine
    return res

def normalize_load(res, h_mean, h_std, c_mean, c_std):
    res.loc[:, (res.columns.str.contains('Heating'))] = (res.loc[:, (res.columns.str.contains('Heating'))] - h_mean) / h_std
    res.loc[:, (res.columns.str.contains('Cooling'))] = (res.loc[:, (res.columns.str.contains('Cooling'))] - c_mean) / c_std
    return res

def read_building_info(file_path):
    bud = pd.read_csv(file_path).dropna(axis=1, how='any')
    bud, _ = scaling_df(bud, start_col=2, type='bud')
    bud.index = bud.id
    return bud

def scaling_df(df, start_col = 0, type = 'cli'):
    if type == 'cli':
        scaler = load(open('data/citydnn_climate_scaler.pkl', 'rb'))
    else:
        scaler = MinMaxScaler()
    try:
        scaled = scaler.fit_transform(df.iloc[:, start_col:])
        df.iloc[:, start_col:] = scaled
    except:
        lab = LabelEncoder()
        for col in df.columns[start_col:]:
            if (df[col].dtype == float or df[col].dtype == int):
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            else:
                df[col] = lab.fit_transform(df[col].values.ravel())
    return df, scaler
