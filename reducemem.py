#!/usr/bin/env python
# coding: utf-8
import gc
import os
import requests
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from numerapi import NumerAPI
from sklearn.preprocessing import PolynomialFeatures
import pyarrow as pa
import pyarrow.parquet as pq
import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
pa.set_cpu_count(4)


def numerai_score(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(
        lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def download_file(url: str, dest_path: str, show_progress_bars: bool = True):
    req = requests.get(url, stream=True)
    req.raise_for_status()
    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))
    if os.path.exists(dest_path):
        file_size = os.stat(dest_path).st_size  # File size in bytes
        if file_size < total_size:
            # Download incomplete
            resume_header = {'Range': 'bytes=%d-' % file_size}
            req = requests.get(url, headers=resume_header, stream=True,
                               verify=False, allow_redirects=True)
        elif file_size == total_size:
            # Download complete
            return
        else:
            # Error, delete file and restart download
            os.remove(dest_path)
            file_size = 0
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(dest_path, "ab") as dest_file:
        for chunk in req.iter_content(block_size):
            progress_bar.update(len(chunk))
            dest_file.write(chunk)

    progress_bar.close()


def download_data(napi, filename, dest_path, roundreq=None):
    query = """
            query ($filename: String!) {
                dataset(filename: $filename)
            }
            """
    params = {
        'filename': filename
    }
    if round:
        query = """
                    query ($filename: String!, $round: Int) {
                        dataset(filename: $filename, round: $round)
                    }
                    """
        params['round'] = roundreq
    dataset_url = napi.raw_query(query, params)['data']['dataset']
    download_file(dataset_url, dest_path, show_progress_bars=True)
    return dataset_url


def CreateMungedParquet(parquetName, mungedParquetName, batch_size, features, otherfeatures, nmf, nmffeatures):
    print('Generating from:',parquetName)
    print('Generating to:',mungedParquetName)
    parquet_file = pq.ParquetFile(parquetName)
    finaldata = None
    totalrecords = parquet_file.metadata.num_rows
    progress_bar = tqdm(total=totalrecords, unit='iB', unit_scale=True)
    for rb in parquet_file.iter_batches(batch_size=batch_size, columns=maincols):
        df = rb.to_pandas().reset_index()
        df[features] = df[features].astype('float32')
        progress_bar.update(df.shape[0])

        chunkotherdata = df[otherfeatures]
        chunkmungeddata = pd.DataFrame(data=np.matmul(df[features].values, nmf),
                                       columns=nmffeatures).astype('float32')
        chunkdata = pd.concat([chunkotherdata, chunkmungeddata], axis=1)
        del chunkotherdata
        del chunkmungeddata
        gc.collect()
        if(finaldata is None):
            finaldata = chunkdata.copy()
        else:
            finaldata = finaldata.append(chunkdata.copy())
        del chunkdata
        gc.collect()
    finaldata.set_index('id', drop=True, inplace=True)
    table = pa.Table.from_pandas(finaldata)
    pq.write_table(table,
                   mungedParquetName,
                   use_dictionary=False,
                   compression='GZIP',
                   compression_level=9,
                   data_page_version="2.0"
                   )

    gc.collect()


napi = NumerAPI()
current_round = napi.get_current_round(tournament=8)
print('Round:', current_round)
if not os.path.isfile(f'./nmfdata/numerai_training_data_{current_round}.parquet'):
    download_data(napi, 'numerai_training_data.parquet',
                  f'./nmfdata/numerai_training_data_{current_round}.parquet', roundreq=current_round)
if not os.path.isfile(f'numerai_tournament_data_{current_round}.parquet'):
    download_data(napi, 'numerai_tournament_data.parquet',
                  f'./nmfdata/numerai_tournament_data_{current_round}.parquet', roundreq=current_round)
if not os.path.isfile(f'numerai_validation_data_{current_round}.parquet'):
    download_data(napi, 'numerai_validation_data.parquet',
                  f'./nmfdata/numerai_validation_data_{current_round}.parquet', roundreq=current_round)
if not os.path.isfile(f'example_predictions_{current_round}.parquet'):
    download_data(napi, 'example_predictions.parquet',
                  f'./nmfdata/example_predictions_{current_round}.parquet', roundreq=current_round)
if not os.path.isfile(f'example_validation_predictions_{current_round}.parquet'):
    download_data(napi, 'example_validation_predictions.parquet', f'./nmfdata/example_validation_predictions_{current_round}.parquet', roundreq=current_round)
maincols = list(np.loadtxt('./nmfdata/traincolumns.txt', dtype='str'))
features = list(maincols[3:-21])
diff = set(maincols) - set(features)
setfeatures = set(features)
otherfeatures = [o for o in maincols if o not in setfeatures]
nmf = np.loadtxt('./nmfdata/LargeKL.csv', delimiter=',')
nmf = nmf.astype('float32')
nmffeatures = ['nmf_'+str(c) for c in range(nmf.shape[1])]
batch_size = 100_000
CreateMungedParquet(f'./nmfdata/numerai_training_data_{current_round}.parquet',
                    f'./nmfdata/numerai_training_data_munged_{current_round}.parquet',
                    batch_size,
                    features,
                    otherfeatures,
                    nmf,
                    nmffeatures)
CreateMungedParquet(f'./nmfdata/numerai_tournament_data_{current_round}.parquet',
                    f'./nmfdata/numerai_tournament_data_munged_{current_round}.parquet',
                    batch_size,
                    features,
                    otherfeatures,
                    nmf,
                    nmffeatures)
CreateMungedParquet(f'./nmfdata/numerai_validation_data_{current_round}.parquet',
                    f'./nmfdata/numerai_validation_data_munged_{current_round}.parquet',
                    batch_size,
                    features,
                    otherfeatures,
                    nmf,
                    nmffeatures)
# Now to compare
df = pd.read_parquet(
    f'./nmfdata/numerai_training_data_{current_round}.parquet')
print(df.info(memory_usage="deep"))

df = pd.read_parquet(
    f'./nmfdata/numerai_training_data_munged_{current_round}.parquet')
print(df.info(memory_usage="deep"))

