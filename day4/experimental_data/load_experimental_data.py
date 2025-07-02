import h5py
from hdf5plugin import Blosc
import numpy as np


def load_i14_merlin(fpath, in_row_of=None):
    """
    Load i14 Merlin data (as a stack).

    Paramters
    ---------
    fpath : str
        the file path
    in_row_of : int, optional
        load as chunks of rows. Default to None, which returns the full
        stack.

    Returns
    -------
    the Merlin stack in np.float32.
    """
    with h5py.File(fpath, "r") as f:
        if in_row_of is None:
            # return all data
            data = f["/entry/merlin_addetector/data"][()]
            return data.reshape(-1, 512, 512).astype(np.float32)
        else:
            # return in chunks
            shape = f["/entry/merlin_addetector/data"].shape
            nr, nc = shape[0], shape[1]
            row_chunks =  nr // in_row_of + 1

            nr_start = 0
            nr_end = in_row_of
            for k in range(row_chunks):
                data = f["/entry/merlin_addetector/data"][nr_start:nr_end, ...]
                yield data.reshape(-1, 512, 512).astype(np.float32)
                nr_start = nr_end
                nr_end += in_row_of


def load_i14_excalibur(fpath, in_chunk_of=None):
    """
    Load i14 Excalibur data.

    Paramters
    ---------
    fpath : str
        the file path
    in_chunk_of : int, optional
        load as chunks of images. Default to None, which returns the full
        stack.

    Returns
    -------
    the Excalibur stack in np.float32.
    """
    with h5py.File(fpath, "r") as f:
        if in_chunk_of is None:
            # return all data
            data = f["/entry/excalibur_addetector/data"][()]
            return data.astype(np.float32)
        else:
            # return in chunks
            shape = f["/entry/excalibur_addetector/data"].shape
            nimg = shape[0]
            nimg_chunks =  nimg // in_chunk_of + 1

            start = 0
            end = in_chunk_of
            for k in range(nimg_chunks):
                data = f["/entry/excalibur_addetector/data"][start:end, ...]
                yield data.astype(np.float32)
                start = end
                end += in_chunk_of
