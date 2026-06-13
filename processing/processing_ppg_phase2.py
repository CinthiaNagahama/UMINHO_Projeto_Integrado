import pandas as pd
import numpy as np

from tqdm import tqdm
import neurokit2 as nk

# Time conversion
FS = 400 # Sample rate of 400Hz (400 samples per second)
MS2MINUTE = 1.6667e-8
MINUTE2MS = 6e7

def import_participants_csv(path, participant):
    df_eventos = pd.read_csv(f'{path}S{participant}/S{participant}_events_tasks.csv', sep=";")
    df_ppg = pd.read_csv(f'{path}S{participant}/S{participant}_PPG_tasks.csv', header = None, names = ['Time', 'PPG'])
    return df_eventos, df_ppg

def _get_code(df_eventos, time):
    code, start, end = df_eventos["label"].values, df_eventos["start_time_ms"].values, df_eventos["end_time_ms"].values
    idx = np.where(np.logical_and(start <= time, end>=time))
    return code[idx][0] if (len(idx[0]) > 0) else None

def _is_end(values):
    return [value.strip() in ('nan', 'end') for value in values]

def prep_eventos(df_eventos):
    df_eventos['Label'] = df_eventos.apply(lambda row: f"B{df_eventos['Label'].iloc[row.name + 1][1:]}" if row['Label'] == 'Recovery' else row['Label'], axis = 1)

    df_eventos.rename({
        'Start_min': 'start_time_min', 
        'End_min': 'end_time_min',
        'Condition': 'condition',
        'Secondary_task': 'secondary_task',
        'Code': 'code',
        'Label': 'label'
    }, axis=1, inplace=True)
    df_eventos['start_time_min'] = [e if isinstance(e, float) else float(e.replace(',', '.')) for e in df_eventos.start_time_min]
    df_eventos['end_time_min'] = [str(e) if isinstance(e, float) else str(e.replace(',', '.')) for e in df_eventos.end_time_min] 
        
    diff_time = [float(t[0]) - t[1] for t in df_eventos[df_eventos['code'] == -1][['end_time_min', 'start_time_min']].iloc[:-1].values]
    df_eventos.loc[_is_end(df_eventos['end_time_min'].str.lower()), 'end_time_min'] = str(df_eventos['start_time_min'].iloc[-1] + np.mean(diff_time))
    df_eventos['label'] = df_eventos['label'].str.strip()
    df_eventos['start_time_ms'] = df_eventos['start_time_min'].apply(lambda t: float(t) * MINUTE2MS)
    df_eventos['end_time_ms'] = df_eventos['end_time_min'].apply(lambda t: float(t) * MINUTE2MS)

    return df_eventos

def gen_lbl2code_dict(df_eventos):
    return {e:c for e, c in df_eventos[['label', 'code']].values}

def prep_ppg(df_ppg, df_eventos):
    df_ppg_processed_nk, _ = nk.ppg_process(df_ppg['PPG'], sampling_rate = FS, method='elgendi')
    df_ppg_processed = df_ppg.join(df_ppg_processed_nk)
    #df_ppg_processed = df_ppg_processed.drop(['PPG', 'PPG_Raw'], axis = 1)
    drop_cols = [c for c in ['PPG', 'PPG_Raw'] if c in df_ppg_processed.columns]
    df_ppg_processed = df_ppg_processed.drop(drop_cols, axis=1)
    df_ppg_processed['label'] = df_ppg_processed['Time'].apply(lambda t: _get_code(df_eventos, t))

    hrv_cols = ['HRV_MeanNN', 'HRV_MedianNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']

    for col in hrv_cols:
        df_ppg_processed[col] = np.nan

    valid_labels = ['B1','B2','B3','B4','B5','B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12','T1','T2','T3','T4','T5','T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']

    for lbl in valid_labels:

        segment = df_ppg_processed[df_ppg_processed['label'] == lbl]
        if segment.empty:
            continue
        segment = segment.reset_index(drop=True)
        peak_idx = segment[segment['PPG_Peaks'] == 1].index.to_numpy()

        if len(peak_idx) < 30:
            continue

        try:
            hrv = nk.hrv(peak_idx, sampling_rate=FS, show=False, nonlinear=False)

            for col in hrv_cols:
                if col in hrv.columns:
                    df_ppg_processed.loc[df_ppg_processed['label'] == lbl, col] = hrv.iloc[0][col]

        except Exception as e:
            print(f'Erro HRV em {lbl}: {e}')
            continue
    
    return df_ppg_processed

def get_baselines(df_ppg):
    return df_ppg[df_ppg['label'].isin(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12'])].copy()

def get_conditions(df_ppg, df_eventos):
    lbl2code = gen_lbl2code_dict(df_eventos)

    df_conditions = df_ppg[df_ppg['label'].isin(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])].copy()
    df_conditions.loc[:, 'code'] = df_conditions['label'].apply(lambda k: lbl2code[k])
    return df_conditions
    
def normalize_baselines_mean_v1(df_conditions, df_baselines, norm_cols):
    # Normalização pela média das baselines
    ppg_baselines_mean = df_baselines.drop('label', axis = 1).mean()
    df_ppg_conditions_norm = df_conditions.copy()
    df_ppg_conditions_norm[norm_cols] = df_ppg_conditions_norm[norm_cols].div(ppg_baselines_mean[norm_cols], axis=1)
    return df_ppg_conditions_norm

def normalize_baselines_mean_v2(df_conditions, df_baselines, norm_cols, log_cols=None, clip_cols=None, epsilon=1e-8):

    df_ppg_conditions_norm = df_conditions.copy()
    baseline = df_baselines.copy()

    if log_cols is not None:

        for col in log_cols:

            if col in df_ppg_conditions_norm.columns:
                df_ppg_conditions_norm[col] = np.log1p(df_ppg_conditions_norm[col].clip(lower=0))

            if col in baseline.columns:
                baseline[col] = np.log1p(baseline[col].clip(lower=0))

    ppg_baselines_mean = baseline[norm_cols].mean()

    for col in norm_cols:

        if (pd.notna(ppg_baselines_mean[col]) and np.isfinite(ppg_baselines_mean[col]) and ppg_baselines_mean[col] != 0):

            df_ppg_conditions_norm[col] = (
                (df_ppg_conditions_norm[col] - ppg_baselines_mean[col]) /
                (ppg_baselines_mean[col] + epsilon)
            )

    if clip_cols is not None:

        for col, limits in clip_cols.items():

            if col in df_ppg_conditions_norm.columns:

                df_ppg_conditions_norm[col] = df_ppg_conditions_norm[col].clip(
                    lower=limits[0],
                    upper=limits[1]
                )
    df_ppg_conditions_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_ppg_conditions_norm

def normalize_baselines_mean(df_conditions, df_baselines, norm_cols):
    # Normalização min-max pela média das baselines
    ppg_baselines_mean = df_baselines.drop('label', axis = 1).mean()
    df_ppg_conditions_norm = df_conditions.copy()

    df_ppg_conditions_norm[norm_cols] = df_ppg_conditions_norm[norm_cols].sub(ppg_baselines_mean[norm_cols]).div(ppg_baselines_mean[norm_cols]) * 100

    min_ppg_norm = df_ppg_conditions_norm[norm_cols].min()
    max_ppg_norm = df_ppg_conditions_norm[norm_cols].max()
    df_ppg_conditions_norm[norm_cols] = (df_ppg_conditions_norm[norm_cols] - min_ppg_norm) / (max_ppg_norm - min_ppg_norm)
    return df_ppg_conditions_norm

def prep_df_final(df_norm, norm_cols):

    grouped = df_norm.groupby('code')

    df_min = grouped[norm_cols].min().add_suffix('_min')
    df_max = grouped[norm_cols].max().add_suffix('_max')
    df_median = grouped[norm_cols].median().add_suffix('_median')
    df_mean = grouped[norm_cols].mean().add_suffix('_mean')
    df_std = grouped[norm_cols].std().fillna(0).add_suffix('_std')
    df_duration = grouped['Time'].agg(lambda x: x.iloc[-1] - x.iloc[0]).to_frame('event_duration_ms')

    df_final = pd.concat([df_min, df_max, df_median, df_mean, df_std, df_duration],axis=1).reset_index()

    return df_final


if __name__ == '__main__':
    dict_conditions = {
        1: 'V1P1',
        2: 'V1P2',
        3: 'V2P1',
        4: 'V2P2',
        5: 'V3P1',
        6: 'V3P2',
        7: 'V1P1',
        8: 'V1P2',
        9: 'V2P1',
        10: 'V2P2',
        11: 'V3P1',
        12: 'V3P2'
    }
    
    df = pd.DataFrame()
    
    # Phase 2
    path_phase_2 = '../../../data/phase_2/raw/'
    participants_phase_2 = ['101', '102', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '115', '117', '119', '120', '121', '122', '123', '124', '125', '128', '129', '130', '131', '132', '133', '134']
                           
    for p in tqdm(participants_phase_2):
        print(f'\nImportando dados do participante S{p}...')
        df_eventos, df_ppg = import_participants_csv(path_phase_2, p)

        print('Preparando datasets...')
        df_eventos = prep_eventos(df_eventos)
        df_ppg = prep_ppg(df_ppg, df_eventos)

        print('Criando tabelas de baselines e conditions...')
        df_baselines = get_baselines(df_ppg)
        df_conditions = get_conditions(df_ppg, df_eventos)

        print('Normalizando os dados...')
        norm_cols = ['PPG_Rate', 'PPG_Quality', 'HRV_MeanNN', 'HRV_MedianNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50']
        df_ppg_norm = normalize_baselines_mean_v2(df_conditions, df_baselines, norm_cols, log_cols=['HRV_LF', 'HRV_HF'], clip_cols=None)
        

        print(f'Gerando o dataset final para o participante S{p}...')
        df_final = prep_df_final(df_ppg_norm, norm_cols)
        df_final['participant'] = f'S{p}'
        df_final['condition'] = [dict_conditions[c] for c in df_final['code']]
        df_final['secondary_task'] = [df_eventos[df_eventos['code'] == c]['secondary_task'].values[0].strip() for c in df_final['code']]

        df = pd.concat([df, df_final], ignore_index=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.dropna(subset=['HRV_RMSSD_mean'], inplace=True)
   
    # Save CSV
    print('Salvando o dataset completo...')
    df.to_csv('../../../data/processed/ppg_p2_v4.csv', index=False)