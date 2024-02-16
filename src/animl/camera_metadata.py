"""
    Camera Metadata Management Module

    This module provides functions and classes for managing files and directories.

    @ Kyra Swanson 2023
"""
import os
from datetime import datetime
import pandas as pd

from . import file_management


def active_times(manifest_dir, depth=1, recursive=True, offset=0):
    """

    """
    # from manifest file
    if file_management.check_file(manifest_dir):
        files = file_management.load_data(manifest_dir)  # load_data(outfile) load file manifest

    # from manifest dataframe
    elif type(manifest_dir) == pd.DataFrame:
        # get time stamps if dne
        if "FileModifyDate" not in manifest_dir.columns:
            files = manifest_dir
            files["FileModifyDate"] = files["FilePath"].apply(lambda x: datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S'))

    # from scratch
    elif os.path.isdir(manifest_dir):
        files = file_management.build_file_manifest(manifest_dir, exif=True, offset=offset,
                                                    recursive=recursive, unique=False)

    else:
        raise FileNotFoundError("Requires a file manifest or image directory.")

    files["Camera"] = files["FilePath"].apply(lambda x: x.split(os.sep)[depth])

    times = files.groupby("Camera").agg({'FileModifyDate': ['min', 'max']})

    return times
