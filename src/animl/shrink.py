"""
    Shrink Module

    Provides functions for dealing with multiple detections/classifications per image

    @ Kyra Swanson 2023
"""
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from numpy import vstack

from . import file_management

pd.set_option('mode.chained_assignment', None)


def shrink(sequence, file_col="FilePath", sort='conf'):

    guesses = sequence.groupby(['prediction'], as_index=False).agg({'confidence': ['max', 'count']})

    guesses = guesses.sort_values(("confidence", "max"), ascending=False).reset_index(drop=True)

    guess = guesses.loc[0, "prediction"].item()
    conf = guesses.loc[0, "confidence"]['max'].item()

    if (guess == "empty") and (len(guesses) > 1):
        guess = guesses.loc[1, "prediction"].item()
        conf = guesses.loc[1, "confidence"]['max'].item()

    sequence = sequence.drop_duplicates(file_col)
    sequence['prediction'] = guess
    sequence['confidence'] = conf

    return sequence


def best_guess(manifest, file_col="FilePath", sort="conf", out_file=None, parallel=False, workers=1):

    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    file_names = manifest[file_col].unique()
    new_df = pd.DataFrame(columns=manifest.columns.values.tolist())

    if parallel:
        pool = mp.Pool(workers)

        new_df = vstack([pool.apply(shrink, args=(manifest, file), kwds={file_col: file_col, sort: sort})
                         for file in tqdm(file_names)])

        pool.close()

    else:
        for i in tqdm(file_names):
            sequence = manifest[manifest[file_col] == i].copy()
            sequence = shrink(sequence)
            new_df = pd.concat([new_df, sequence])

    if out_file is not None:
        file_management.save_data(new_df, out_file)

    return new_df
