{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4af7ca5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import gc\n",
    "import os\n",
    "import requests\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from termcolor import colored\n",
    "from tqdm.notebook import tqdm\n",
    "from numerapi import NumerAPI\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "import pyarrow as pa\n",
    "import pyarrow.parquet as pq\n",
    "import os\n",
    "os.environ['NUMEXPR_MAX_THREADS'] = '8'\n",
    "os.environ['NUMEXPR_NUM_THREADS'] = '4'\n",
    "pa.set_cpu_count(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7372e14b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def numerai_score(y_true, y_pred, eras):\n",
    "    rank_pred = y_pred.groupby(eras).apply(\n",
    "        lambda x: x.rank(pct=True, method=\"first\"))\n",
    "    return np.corrcoef(y_true, rank_pred)[0, 1]\n",
    "\n",
    "\n",
    "def correlation_score(y_true, y_pred):\n",
    "    return np.corrcoef(y_true, y_pred)[0, 1]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "4b802eeb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def download_file(url: str, dest_path: str, show_progress_bars: bool = True):\n",
    "\n",
    "    req = requests.get(url, stream=True)\n",
    "    req.raise_for_status()\n",
    "\n",
    "    # Total size in bytes.\n",
    "    total_size = int(req.headers.get('content-length', 0))\n",
    "    if os.path.exists(dest_path):\n",
    "        file_size = os.stat(dest_path).st_size  # File size in bytes\n",
    "        if file_size < total_size:\n",
    "            # Download incomplete\n",
    "            resume_header = {'Range': 'bytes=%d-' % file_size}\n",
    "            req = requests.get(url, headers=resume_header, stream=True,\n",
    "                               verify=False, allow_redirects=True)\n",
    "        elif file_size == total_size:\n",
    "            # Download complete\n",
    "            return\n",
    "        else:\n",
    "            # Error, delete file and restart download\n",
    "            os.remove(dest_path)\n",
    "            file_size = 0\n",
    "\n",
    "    response = requests.get(url, stream=True)\n",
    "    total_size_in_bytes= int(response.headers.get('content-length', 0))\n",
    "    block_size = 1024 #1 Kibibyte\n",
    "    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)\n",
    "    \n",
    "    \n",
    "    \n",
    "    with open(dest_path, \"ab\") as dest_file:\n",
    "        for chunk in req.iter_content(block_size):\n",
    "            progress_bar.update(len(chunk))\n",
    "            dest_file.write(chunk)\n",
    "            \n",
    "    progress_bar.close()\n",
    "\n",
    "\n",
    "def download_data(napi, filename, dest_path, roundreq=None):\n",
    "    query = \"\"\"\n",
    "            query ($filename: String!) {\n",
    "                dataset(filename: $filename)\n",
    "            }\n",
    "            \"\"\"\n",
    "    params = {\n",
    "        'filename': filename\n",
    "    }\n",
    "    if round:\n",
    "        query = \"\"\"\n",
    "                    query ($filename: String!, $round: Int) {\n",
    "                        dataset(filename: $filename, round: $round)\n",
    "                    }\n",
    "                    \"\"\"\n",
    "        params['round'] = roundreq\n",
    "    dataset_url = napi.raw_query(query, params)['data']['dataset']\n",
    "    download_file(dataset_url, dest_path, show_progress_bars=True)\n",
    "    return dataset_url\n",
    "\n",
    "\n",
    "\n",
    "def CreateMungedParquet(parquetName,mungedParquetName,batch_size,features,otherfeatures,nmf,nmffeatures):\n",
    "       \n",
    "    parquet_file = pq.ParquetFile(parquetName)\n",
    "    finaldata = None\n",
    "    totalrecords = parquet_file.metadata.num_rows\n",
    "    progress_bar = tqdm(total=totalrecords, unit='iB', unit_scale=True)\n",
    "\n",
    "    for rb in parquet_file.iter_batches(batch_size=batch_size, columns=maincols):\n",
    "        df = rb.to_pandas().reset_index()\n",
    "        df[features] = df[features].astype('float32')\n",
    "        progress_bar.update(df.shape[0])\n",
    "        \n",
    "        \n",
    "        chunkotherdata = df[otherfeatures]\n",
    "        chunkmungeddata = pd.DataFrame(data=np.matmul(df[features].values,nmf),\n",
    "                                       columns=nmffeatures).astype('float32')\n",
    "        chunkdata = pd.concat([chunkotherdata,chunkmungeddata],axis=1)\n",
    "        del chunkotherdata\n",
    "        del chunkmungeddata\n",
    "        gc.collect()\n",
    "        if(finaldata is None):\n",
    "            finaldata = chunkdata.copy()\n",
    "        else:\n",
    "            finaldata = finaldata.append(chunkdata.copy())\n",
    "        del chunkdata\n",
    "        gc.collect()\n",
    "\n",
    "    finaldata.set_index('id',drop=True,inplace=True)\n",
    "    table = pa.Table.from_pandas(finaldata)\n",
    "    finaldata.to_parquet(mungedParquetName,compression='gzip')\n",
    "    del finaldata\n",
    "    pq.write_table(table,\n",
    "               mungedParquetName,\n",
    "               use_dictionary=False,\n",
    "               compression='GZIP',\n",
    "               compression_level=9,\n",
    "               data_page_version = \"2.0\"    \n",
    "              )\n",
    "    \n",
    "    gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "71dfc6c0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Round: 297\n"
     ]
    }
   ],
   "source": [
    "napi = NumerAPI()\n",
    "current_round = napi.get_current_round(tournament=8)\n",
    "print('Round:',current_round)\n",
    "if not os.path.isfile(f'./nmfdata/numerai_training_data_{current_round}.parquet'):\n",
    "    download_data(napi, 'numerai_training_data.parquet', f'./nmfdata/numerai_training_data_{current_round}.parquet', roundreq=current_round)\n",
    "if not os.path.isfile(f'numerai_tournament_data_{current_round}.parquet'):\n",
    "    download_data(napi, 'numerai_tournament_data.parquet', f'./nmfdata/numerai_tournament_data_{current_round}.parquet', roundreq=current_round)\n",
    "if not os.path.isfile(f'numerai_validation_data_{current_round}.parquet'):\n",
    "    download_data(napi, 'numerai_validation_data.parquet', f'./nmfdata/numerai_validation_data_{current_round}.parquet', roundreq=current_round)\n",
    "if not os.path.isfile(f'example_predictions_{current_round}.parquet'):\n",
    "    download_data(napi, 'example_predictions.parquet', f'./nmfdata/example_predictions_{current_round}.parquet', roundreq=current_round)\n",
    "if not os.path.isfile(f'example_validation_predictions_{current_round}.parquet'):\n",
    "    download_data(napi, 'example_validation_predictions.parquet', f'./nmfdata/example_validation_predictions_{current_round}.parquet', roundreq=current_round)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5b980252",
   "metadata": {},
   "outputs": [],
   "source": [
    "maincols = list(np.loadtxt('./nmfdata/traincolumns.txt',dtype='str'))\n",
    "features  = list(maincols[3:-21])\n",
    "diff = set(maincols) - set(features)\n",
    "setfeatures = set(features)\n",
    "otherfeatures = [o for o in maincols if o not in setfeatures]\n",
    "nmf = np.loadtxt('./nmfdata/LargeKL.csv', delimiter=',')\n",
    "nmf = nmf.astype('float32')\n",
    "nmffeatures = ['nmf_'+str(c) for c in range(nmf.shape[1])]\n",
    "batch_size=100_000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "9ca1ca32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b8b264bad26477a9e24d71fdf8fe627",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0.00/2.41M [00:00<?, ?iB/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "CreateMungedParquet(f'./nmfdata/numerai_training_data_{current_round}.parquet',\n",
    "                f'./nmfdata/numerai_training_data_munged_{current_round}.parquet',\n",
    "                batch_size,\n",
    "                features,\n",
    "                otherfeatures,\n",
    "                nmf,\n",
    "                nmffeatures)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "db20a077",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7a5bf91e16c4421fa24e4a1e527bd20c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0.00/1.41M [00:00<?, ?iB/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "CreateMungedParquet(f'./nmfdata/numerai_tournament_data_{current_round}.parquet',\n",
    "                f'./nmfdata/numerai_tournament_data_munged_{current_round}.parquet',\n",
    "                batch_size,\n",
    "                features,\n",
    "                otherfeatures,\n",
    "                nmf,\n",
    "                nmffeatures)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "cf012ca2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "91ad8f65cc9a467b8db7aac676f6ccd2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0.00/540k [00:00<?, ?iB/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "CreateMungedParquet(f'./nmfdata/numerai_validation_data_{current_round}.parquet',\n",
    "                f'./nmfdata/numerai_validation_data_munged_{current_round}.parquet',\n",
    "                batch_size,\n",
    "                features,\n",
    "                otherfeatures,\n",
    "                nmf,\n",
    "                nmffeatures)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9d9ff43b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now to compare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "e71a7e05",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 2412105 entries, n003bba8a98662e4 to nfff2bd38e397265\n",
      "Columns: 1073 entries, era to target_thomas_60\n",
      "dtypes: float32(1071), object(2)\n",
      "memory usage: 10.1 GB\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_parquet(f'./nmfdata/numerai_training_data_{current_round}.parquet')\n",
    "df.info(memory_usage=\"deep\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "46b5cb94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>era</th>\n",
       "      <th>data_type</th>\n",
       "      <th>feature_dichasial_hammier_spawner</th>\n",
       "      <th>feature_rheumy_epistemic_prancer</th>\n",
       "      <th>feature_pert_performative_hormuz</th>\n",
       "      <th>feature_hillier_unpitied_theobromine</th>\n",
       "      <th>feature_perigean_bewitching_thruster</th>\n",
       "      <th>feature_renegade_undomestic_milord</th>\n",
       "      <th>feature_koranic_rude_corf</th>\n",
       "      <th>feature_demisable_expiring_millepede</th>\n",
       "      <th>...</th>\n",
       "      <th>target_paul_20</th>\n",
       "      <th>target_paul_60</th>\n",
       "      <th>target_george_20</th>\n",
       "      <th>target_george_60</th>\n",
       "      <th>target_william_20</th>\n",
       "      <th>target_william_60</th>\n",
       "      <th>target_arthur_20</th>\n",
       "      <th>target_arthur_60</th>\n",
       "      <th>target_thomas_20</th>\n",
       "      <th>target_thomas_60</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>n003bba8a98662e4</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.50</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>...</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.166667</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.166667</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.166667</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n003bee128c2fcfc</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.5</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.75</td>\n",
       "      <td>...</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.833333</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.833333</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.833333</td>\n",
       "      <td>0.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n0048ac83aff7194</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>...</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.333333</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.333333</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n00691bec80d3e02</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.25</td>\n",
       "      <td>1.00</td>\n",
       "      <td>...</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n00b8720a2fdc4f2</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.50</td>\n",
       "      <td>...</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 1073 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                   era data_type  feature_dichasial_hammier_spawner  \\\n",
       "id                                                                    \n",
       "n003bba8a98662e4  0001     train                                1.0   \n",
       "n003bee128c2fcfc  0001     train                                0.5   \n",
       "n0048ac83aff7194  0001     train                                0.5   \n",
       "n00691bec80d3e02  0001     train                                1.0   \n",
       "n00b8720a2fdc4f2  0001     train                                1.0   \n",
       "\n",
       "                  feature_rheumy_epistemic_prancer  \\\n",
       "id                                                   \n",
       "n003bba8a98662e4                              0.50   \n",
       "n003bee128c2fcfc                              1.00   \n",
       "n0048ac83aff7194                              0.25   \n",
       "n00691bec80d3e02                              0.50   \n",
       "n00b8720a2fdc4f2                              0.75   \n",
       "\n",
       "                  feature_pert_performative_hormuz  \\\n",
       "id                                                   \n",
       "n003bba8a98662e4                              1.00   \n",
       "n003bee128c2fcfc                              0.25   \n",
       "n0048ac83aff7194                              0.75   \n",
       "n00691bec80d3e02                              0.50   \n",
       "n00b8720a2fdc4f2                              1.00   \n",
       "\n",
       "                  feature_hillier_unpitied_theobromine  \\\n",
       "id                                                       \n",
       "n003bba8a98662e4                                  1.00   \n",
       "n003bee128c2fcfc                                  0.75   \n",
       "n0048ac83aff7194                                  0.00   \n",
       "n00691bec80d3e02                                  0.75   \n",
       "n00b8720a2fdc4f2                                  1.00   \n",
       "\n",
       "                  feature_perigean_bewitching_thruster  \\\n",
       "id                                                       \n",
       "n003bba8a98662e4                                  0.00   \n",
       "n003bee128c2fcfc                                  0.00   \n",
       "n0048ac83aff7194                                  0.75   \n",
       "n00691bec80d3e02                                  0.00   \n",
       "n00b8720a2fdc4f2                                  0.00   \n",
       "\n",
       "                  feature_renegade_undomestic_milord  \\\n",
       "id                                                     \n",
       "n003bba8a98662e4                                0.00   \n",
       "n003bee128c2fcfc                                0.75   \n",
       "n0048ac83aff7194                                0.00   \n",
       "n00691bec80d3e02                                1.00   \n",
       "n00b8720a2fdc4f2                                0.00   \n",
       "\n",
       "                  feature_koranic_rude_corf  \\\n",
       "id                                            \n",
       "n003bba8a98662e4                       1.00   \n",
       "n003bee128c2fcfc                       0.50   \n",
       "n0048ac83aff7194                       0.75   \n",
       "n00691bec80d3e02                       0.25   \n",
       "n00b8720a2fdc4f2                       1.00   \n",
       "\n",
       "                  feature_demisable_expiring_millepede  ...  target_paul_20  \\\n",
       "id                                                      ...                   \n",
       "n003bba8a98662e4                                  1.00  ...            0.25   \n",
       "n003bee128c2fcfc                                  0.75  ...            1.00   \n",
       "n0048ac83aff7194                                  0.75  ...            0.50   \n",
       "n00691bec80d3e02                                  1.00  ...            0.50   \n",
       "n00b8720a2fdc4f2                                  0.50  ...            0.50   \n",
       "\n",
       "                  target_paul_60  target_george_20  target_george_60  \\\n",
       "id                                                                     \n",
       "n003bba8a98662e4            0.25              0.25              0.00   \n",
       "n003bee128c2fcfc            1.00              1.00              1.00   \n",
       "n0048ac83aff7194            0.25              0.25              0.25   \n",
       "n00691bec80d3e02            0.50              0.50              0.50   \n",
       "n00b8720a2fdc4f2            0.50              0.50              0.50   \n",
       "\n",
       "                  target_william_20  target_william_60  target_arthur_20  \\\n",
       "id                                                                         \n",
       "n003bba8a98662e4           0.166667           0.000000          0.166667   \n",
       "n003bee128c2fcfc           0.833333           0.666667          0.833333   \n",
       "n0048ac83aff7194           0.500000           0.333333          0.500000   \n",
       "n00691bec80d3e02           0.666667           0.500000          0.500000   \n",
       "n00b8720a2fdc4f2           0.666667           0.500000          0.500000   \n",
       "\n",
       "                  target_arthur_60  target_thomas_20  target_thomas_60  \n",
       "id                                                                      \n",
       "n003bba8a98662e4          0.000000          0.166667          0.000000  \n",
       "n003bee128c2fcfc          0.666667          0.833333          0.666667  \n",
       "n0048ac83aff7194          0.333333          0.500000          0.333333  \n",
       "n00691bec80d3e02          0.500000          0.666667          0.500000  \n",
       "n00b8720a2fdc4f2          0.500000          0.666667          0.500000  \n",
       "\n",
       "[5 rows x 1073 columns]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "d1e87fd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 2412105 entries, n003bba8a98662e4 to nfff2bd38e397265\n",
      "Columns: 151 entries, era to nmf_127\n",
      "dtypes: float32(149), object(2)\n",
      "memory usage: 1.8 GB\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_parquet(f'./nmfdata/numerai_training_data_munged_{current_round}.parquet')\n",
    "df.info(memory_usage=\"deep\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "479f82bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>era</th>\n",
       "      <th>data_type</th>\n",
       "      <th>target</th>\n",
       "      <th>target_nomi_20</th>\n",
       "      <th>target_nomi_60</th>\n",
       "      <th>target_jerome_20</th>\n",
       "      <th>target_jerome_60</th>\n",
       "      <th>target_janet_20</th>\n",
       "      <th>target_janet_60</th>\n",
       "      <th>target_ben_20</th>\n",
       "      <th>...</th>\n",
       "      <th>nmf_118</th>\n",
       "      <th>nmf_119</th>\n",
       "      <th>nmf_120</th>\n",
       "      <th>nmf_121</th>\n",
       "      <th>nmf_122</th>\n",
       "      <th>nmf_123</th>\n",
       "      <th>nmf_124</th>\n",
       "      <th>nmf_125</th>\n",
       "      <th>nmf_126</th>\n",
       "      <th>nmf_127</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>n003bba8a98662e4</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>...</td>\n",
       "      <td>3.758544</td>\n",
       "      <td>3.347078</td>\n",
       "      <td>3.405092</td>\n",
       "      <td>3.457315</td>\n",
       "      <td>3.365403</td>\n",
       "      <td>3.504374</td>\n",
       "      <td>3.464564</td>\n",
       "      <td>3.522398</td>\n",
       "      <td>3.569177</td>\n",
       "      <td>3.550374</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n003bee128c2fcfc</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.75</td>\n",
       "      <td>1.00</td>\n",
       "      <td>...</td>\n",
       "      <td>4.020459</td>\n",
       "      <td>3.753796</td>\n",
       "      <td>3.845602</td>\n",
       "      <td>3.826428</td>\n",
       "      <td>3.821350</td>\n",
       "      <td>3.834328</td>\n",
       "      <td>3.769660</td>\n",
       "      <td>3.887188</td>\n",
       "      <td>3.831103</td>\n",
       "      <td>3.805821</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n0048ac83aff7194</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.50</td>\n",
       "      <td>...</td>\n",
       "      <td>3.928508</td>\n",
       "      <td>3.571933</td>\n",
       "      <td>3.733849</td>\n",
       "      <td>3.728535</td>\n",
       "      <td>3.800013</td>\n",
       "      <td>3.732891</td>\n",
       "      <td>3.653759</td>\n",
       "      <td>3.735944</td>\n",
       "      <td>3.715369</td>\n",
       "      <td>3.851631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n00691bec80d3e02</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>...</td>\n",
       "      <td>3.638891</td>\n",
       "      <td>3.298438</td>\n",
       "      <td>3.487785</td>\n",
       "      <td>3.526153</td>\n",
       "      <td>3.595376</td>\n",
       "      <td>3.572592</td>\n",
       "      <td>3.660121</td>\n",
       "      <td>3.574319</td>\n",
       "      <td>3.591986</td>\n",
       "      <td>3.645196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>n00b8720a2fdc4f2</th>\n",
       "      <td>0001</td>\n",
       "      <td>train</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.50</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.50</td>\n",
       "      <td>...</td>\n",
       "      <td>3.588022</td>\n",
       "      <td>3.393866</td>\n",
       "      <td>3.499561</td>\n",
       "      <td>3.543274</td>\n",
       "      <td>3.461243</td>\n",
       "      <td>3.618092</td>\n",
       "      <td>3.471658</td>\n",
       "      <td>3.522941</td>\n",
       "      <td>3.659844</td>\n",
       "      <td>3.564431</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 151 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                   era data_type  target  target_nomi_20  target_nomi_60  \\\n",
       "id                                                                         \n",
       "n003bba8a98662e4  0001     train    0.25            0.25            0.00   \n",
       "n003bee128c2fcfc  0001     train    0.75            0.75            0.75   \n",
       "n0048ac83aff7194  0001     train    0.50            0.50            0.25   \n",
       "n00691bec80d3e02  0001     train    0.75            0.75            0.50   \n",
       "n00b8720a2fdc4f2  0001     train    0.75            0.75            0.50   \n",
       "\n",
       "                  target_jerome_20  target_jerome_60  target_janet_20  \\\n",
       "id                                                                      \n",
       "n003bba8a98662e4              0.25              0.25             0.25   \n",
       "n003bee128c2fcfc              1.00              0.75             1.00   \n",
       "n0048ac83aff7194              0.50              0.25             0.25   \n",
       "n00691bec80d3e02              0.50              0.50             0.50   \n",
       "n00b8720a2fdc4f2              0.50              0.50             0.75   \n",
       "\n",
       "                  target_janet_60  target_ben_20  ...   nmf_118   nmf_119  \\\n",
       "id                                                ...                       \n",
       "n003bba8a98662e4             0.25           0.25  ...  3.758544  3.347078   \n",
       "n003bee128c2fcfc             0.75           1.00  ...  4.020459  3.753796   \n",
       "n0048ac83aff7194             0.25           0.50  ...  3.928508  3.571933   \n",
       "n00691bec80d3e02             0.75           0.75  ...  3.638891  3.298438   \n",
       "n00b8720a2fdc4f2             0.75           0.50  ...  3.588022  3.393866   \n",
       "\n",
       "                   nmf_120   nmf_121   nmf_122   nmf_123   nmf_124   nmf_125  \\\n",
       "id                                                                             \n",
       "n003bba8a98662e4  3.405092  3.457315  3.365403  3.504374  3.464564  3.522398   \n",
       "n003bee128c2fcfc  3.845602  3.826428  3.821350  3.834328  3.769660  3.887188   \n",
       "n0048ac83aff7194  3.733849  3.728535  3.800013  3.732891  3.653759  3.735944   \n",
       "n00691bec80d3e02  3.487785  3.526153  3.595376  3.572592  3.660121  3.574319   \n",
       "n00b8720a2fdc4f2  3.499561  3.543274  3.461243  3.618092  3.471658  3.522941   \n",
       "\n",
       "                   nmf_126   nmf_127  \n",
       "id                                    \n",
       "n003bba8a98662e4  3.569177  3.550374  \n",
       "n003bee128c2fcfc  3.831103  3.805821  \n",
       "n0048ac83aff7194  3.715369  3.851631  \n",
       "n00691bec80d3e02  3.591986  3.645196  \n",
       "n00b8720a2fdc4f2  3.659844  3.564431  \n",
       "\n",
       "[5 rows x 151 columns]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73216935",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
