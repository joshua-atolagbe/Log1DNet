import numpy as np
import pandas as pd
import lasio
import os

#select columns to use in the dataframe
col = ['NPHI', 'RHOB', 'GR', 'RT','DT', 'DTS', 'DEPTH']
f_cols = ['DEPTH', 'WELL', 'GR', 'RT', 'RHOB', 'NPHI', 'DTC', 'DTS']

def process_train(well:pd.DataFrame, cols:list, name:str) -> pd.DataFrame:

    '''
    cleans and process the train dataframe for null values

    '''
#     column = cols[:-2]
    df = well.filter(cols, axis='columns')
    df = df.dropna(axis='index',
                   subset=['NPHI','DTS', 'DT']).rename({'DT':'DTC'},
                                                       axis='columns').reset_index(drop=True).sort_values('DEPTH')
    df['WELL'] = name

    return df[f_cols]

def process_test(well:pd.DataFrame, cols:list, name:str) -> pd.DataFrame:

    '''
    cleans and process the test dataframe for null values

    '''
#     column = cols[:-2]
    if name == '15_9-F-5' or name == '15_9-F-15D':
        column = ['NPHI', 'RHOB', 'GR', 'RT', 'DEPTH']
        df = well.filter(column, axis='columns')
        df = df.dropna(axis='index', subset=['NPHI']).reset_index(drop=True).sort_values('DEPTH')
        # df_shape = df.shape
        df['WELL'] = name

        if name =='15_9-F-5':
            df['DEPTH'] = round((df['DEPTH']*0.0254)/10, 1)

        return df[f_cols[:6]]

    else:
        df = well.filter(cols, axis='columns')
        df = df.dropna(axis='index',
                       subset=['DTS', 'DT']).rename({'DT':'DTC'},
                                                           axis='columns').reset_index(drop=True).sort_values('DEPTH')
        df['WELL'] = name
        df['NPHI'] = np.where(df['NPHI'] < 0, np.nan, df['NPHI'])
        df = df.dropna().reset_index(drop=True)#well2
        if name == '15_9-F-14' or name == '15_9-F-4':

            df['DEPTH'] = round((df['DEPTH']*0.0254)/10, 1)


        return df[f_cols]
