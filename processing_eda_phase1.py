import pandas as pd
import numpy as np

import neurokit2 as nk

# Time conversion
FS = 400 # Sample rate of 400Hz (400 samples per second)
MS2MINUTE = 1.6667e-8
MINUTE2MS = 6e7

def import_participants_csv(path, participant):
    df_eventos = pd.read_csv(f'{path}S{participant}/events_S{participant}.csv')
    df_eda = pd.read_csv(f'{path}S{participant}/S{participant}_EDA_tasks.csv')
    return df_eventos, df_eda

def _get_code(df_eventos, time):
    code, start, end = df_eventos["label"].values, df_eventos["start_time_ms"].values, df_eventos["end_time_ms"].values
    idx = np.where(np.logical_and(start <= time, end>=time))
    return code[idx][0] if (len(idx[0]) > 0) else None

def prep_eventos(df_eventos, df_performance):
    df_eventos.rename({
        'Biopac (min)': 'start_time_min', 
        'Biopac-end(min)': 'end_time_min',
        'Condition': 'condition',
        'Secondary task': 'secondary_task',
        'Code': 'code'
    }, axis=1, inplace=True)
    
    diff_time = [float(t[0]) - t[1] for t in df_eventos[df_eventos['code'] == 'Questionario'][['end_time_min', 'start_time_min']].iloc[:-1].values]
    df_eventos.loc[df_eventos['end_time_min'].str.lower() == 'end', 'end_time_min'] = str(df_eventos['start_time_min'].iloc[-1] + np.mean(diff_time))
    df_eventos['label'] = df_eventos['label'].str.strip()
    df_eventos['start_time_ms'] = [float(t) * MINUTE2MS for t in df_eventos['start_time_min']]
    df_eventos['end_time_ms'] = [float(t) * MINUTE2MS for t in df_eventos['end_time_min']]

    df_aux = pd.merge(
        left=df_eventos, 
        right=df_performance[['Nº', 'Sequence', 'Errors', 'Errors rate']], 
        how='left',
        left_on=['user', 'condition'], 
        right_on=['Nº', 'Sequence']
    )
    df_aux.rename({'Errors':'errors', 'Errors rate':'errors_rate'}, axis=1, inplace=True)
    df_aux.drop(['Nº', 'Sequence'], axis=1, inplace=True)
    return df_aux

def gen_lbl2code_dict(df_eventos):
    return {e:c for e, c in df_eventos[['label', 'code']].values}

def prep_eda(df_eda, df_eventos):
    df_eda_processed_nk, _ = nk.eda_process(df_eda['EDA'], sampling_rate = FS, method='neurokit')
    df_eda_processed = df_eda.join(df_eda_processed_nk)
    df_eda_processed = df_eda_processed.drop(['EDA', 'EDA_Raw'], axis = 1)
    df_eda_processed['label'] = df_eda_processed['Time'].apply(lambda t: _get_code(df_eventos, t))
    df_eda_processed = pd.merge(left=df_eda_processed, right=df_eventos[['label', 'errors', 'errors_rate']], how='left', on='label')
    return df_eda_processed

def get_baselines(df_eda):
    return df_eda[df_eda['label'].isin(['B1', 'B2', 'B3', 'B4', 'B5', 'B6'])].copy()

def get_conditions(df_eda, df_eventos):
    lbl2code = gen_lbl2code_dict(df_eventos)

    df_conditions = df_eda[df_eda['label'].isin(['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])].copy()
    df_conditions.loc[:, 'code'] = df_conditions['label'].apply(lambda k: lbl2code[k])
    return df_conditions

def normalize_baselines_mean(df_conditions, df_baselines, norm_cols):
    # Normalização min-max pela média das baselines
    eda_baselines_mean = df_baselines.drop('label', axis = 1).mean()
    df_eda_conditions_norm = df_conditions.copy()

    df_eda_conditions_norm[norm_cols] = df_eda_conditions_norm[norm_cols].sub(eda_baselines_mean[norm_cols]).div(eda_baselines_mean[norm_cols]) * 100

    min_eda_norm = df_eda_conditions_norm[norm_cols].min()
    max_eda_norm = df_eda_conditions_norm[norm_cols].max()
    df_eda_conditions_norm[norm_cols] = (df_eda_conditions_norm[norm_cols] - min_eda_norm) / (max_eda_norm - min_eda_norm)
    return df_eda_conditions_norm

def prep_df_final(df_norm, norm_cols):
    df_final = df_norm.groupby('code')[['SCR_Onsets', 'SCR_Peaks', 'SCR_Recovery']].sum()

    df_final[norm_cols] = df_norm.groupby('code')[norm_cols].min()
    df_final.rename({n: n + '_min' for n in norm_cols}, inplace = True, axis = 1)
    
    df_final[norm_cols] = df_norm.groupby('code')[norm_cols].max()
    df_final.rename({n: n + '_max' for n in norm_cols}, inplace = True, axis = 1)
    
    df_final[norm_cols] = df_norm.groupby('code')[norm_cols].median()
    df_final.rename({n: n + '_median' for n in norm_cols}, inplace = True, axis = 1)
    
    df_final[norm_cols] = df_norm.groupby('code')[norm_cols].mean()
    df_final.rename({n: n + '_mean' for n in norm_cols}, inplace = True, axis = 1)
    
    df_final[norm_cols] = df_norm.groupby('code')[norm_cols].std()
    df_final.rename({n: n + '_std' for n in norm_cols}, inplace = True, axis = 1)

    df_final[['errors', 'errors_rate']] = df_norm.groupby('code')[['errors', 'errors_rate']].max()
    df_final['event_duration_ms'] = df_norm.groupby('code')['Time'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    df_final = df_final.reset_index()
    return df_final


if __name__ == '__main__':
    dict_conditions = {
        'C1': 'V1P1',
        'C2': 'V1P2',
        'C3': 'V2P1',
        'C4': 'V2P2',
        'C5': 'V3P1',
        'C6': 'V3P2'
    }
    
    df = pd.DataFrame()
    
    # Phase 1
    path_phase_1 = f'../../data/phase_1_v2/raw/'
    participants_phase_1 = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '20', '21', '22', '23', '24', '26', '27', '28', '29', '30']
    
    print('\nImportando dados de performance dos participantes')
    df_performance = pd.read_excel(f'../../data/phase_1_v2/raw/performance data/WKL_fase1_PerformanceData_ERRORS.xlsx')
    df_performance['Nº'] = df_performance['Nº'].ffill()

    for p in participants_phase_1:
        print(f'\nImportando dados do participante S{p}...')
        df_eventos, df_eda = import_participants_csv(path_phase_1, p)

        print('Preparando datasets...')
        df_eventos = prep_eventos(df_eventos, df_performance)
        df_eda = prep_eda(df_eda, df_eventos)

        print('Criando tabelas de baselines e conditions...')
        df_baselines = get_baselines(df_eda)
        df_conditions = get_conditions(df_eda, df_eventos)

        print('Normalizando os dados...')
        norm_cols = ['EDA_Clean', 'EDA_Tonic', 'EDA_Phasic', 'SCR_Height', 'SCR_Amplitude', 'SCR_RiseTime', 'SCR_RecoveryTime']
        df_eda_norm = normalize_baselines_mean(df_conditions, df_baselines, norm_cols)
        

        print(f'Gerando o dataset final para o participante S{p}...')
        df_final = prep_df_final(df_eda_norm, norm_cols)
        df_final['participant'] = f'S{p}'
        df_final['condition'] = [dict_conditions[c] for c in df_final['code']]
        df_final['secondary_task'] = 'yes'

        df = pd.concat([df, df_final])

    # Save CSV
    print('Salvando o dataset completo...')
    df.to_csv('../../data/processed/eda_normalizado_p1v3.csv')