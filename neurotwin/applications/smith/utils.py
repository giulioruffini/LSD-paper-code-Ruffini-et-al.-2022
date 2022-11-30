"""
SMITH
UTILITIES
-------------------------------------------------------------------------------
This module is used to binarized fMRI or EEG data before using the inet
and itailor module. It takes an array of 2-D data (EEF or fMRI) with
Regions Of Interest (ROIs) (or nodes, or electrodes) x time, and converts
it to an array of the same dimension containing the binarized version
of the original data.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


# - BINARIZATION --------------------------------------------------------------
def binarize(data: np.ndarray,
             threshold_method: str) -> np.ndarray:
    """
    The function takes the vector data, which should be in format (ROIs, time)
    and binarizes it given a threshold. The function uses the specified method
    i.e. 'mean', 'median' or 'std' to compute the numeric threshold (per ROI)
    that will split which data will be valued -1 or 1. Data falling exactly at
    the threshold value is assigned 1 or -1 randomly.

    Args:
        data: DIMS: (ROIs, time points), UNITS: arbitrary. Data units
            are arbitrary (as long as they are consistent across subjects and
            sessions) since the Ising model has no units and performs a
            qualitative analysis of the data.
        threshold_method: method to be used to calculate the threshold.
            It has to be either 'mean', 'median' or 'std'.

    Returns:
        A binarized data array. Has the same dimensions as the data and has the
        same information that the data but in a binarized way, only with 1.0
        and -1.0.

    """
    # Get number of nodes and length of data
    node_number: int
    data_length: int
    [node_number, data_length] = np.shape(data)
    # Create vector with threshold values depending on threshold method chosen
    thresh_vec: np.ndarray = np.empty([node_number, 1], dtype=float)
    if threshold_method.casefold() == 'mean':
        thresh_vec = np.reshape(np.mean(data, 1), ([node_number, 1]))
    elif threshold_method.casefold() == "median":
        thresh_vec = np.reshape(np.median(data, 1), ([node_number, 1]))
    elif threshold_method.casefold() == "std":
        thresh_vec = np.reshape(np.std(data, 1), ([node_number, 1]))
        average: np.ndarray = np.reshape(np.mean(data, 1),
                                         [node_number, 1])
        data = data - average * np.ones([1, data_length])
    # Create threshold matrix of size of data array
    thresh_mat: np.ndarray = np.multiply(thresh_vec, np.ones([node_number,
                                                              data_length]))
    # Assign +1 (-1) to values greater (smaller) than threshold
    binarized_data: np.ndarray = np.sign(data - thresh_mat)
    # Do the following to handle cases where binarization yields zeros
    # Count the number of zeros found
    num_zeros: int = int(np.sum(binarized_data == 0))
    # Safety checks: store number of zeros, threshold method and matrices in
    # the logger
    _msg: str = f"Binarizer found {num_zeros} zeros. Threshold method " \
                f"was {threshold_method}. Threshold vectors were \n" \
                f"{thresh_vec}"
    logger.info(_msg)
    # Randomly assign +1 or -1 to zeros
    binarized_data[binarized_data == 0] = np.random.choice([-1, 1], num_zeros)
    return binarized_data


# - STARLAB TN0344 METHODS ----------------------------------------------------
def lzw_compress(data_string: str,
                 mode: str = 'binary',
                 verbose: bool = False):
    """
    Compress a string to a list of output symbols using the
    Lempel-Ziv-Welch (LZW) compression.

    The current method works by reading a sequence of symbols,
    grouping the symbols into strings and converting the strings
    into codes. Starts from two symbols, 0 and 1.

    Args:
        data_string: string ot compress.
        mode: either binary or ascii.
        verbose: boolean to indicate whether additional information
            wants to be displayed through the terminal.
    Returns:
        the compressed string and the length of the dictionary
        If you need to, convert first arrays to a string,
        e.g., entry="".join([np.str(np.int(x)) for x in data_array])
    """

    if mode == 'binary':
        dict_size: int = 2
        dictionary: dict = {'0': 0, '1': 1}
    elif mode == 'ascii':
        # Build the dictionary for generic ascii.
        dict_size: int = 256
        dictionary: dict = dict((chr(i), i) for i in range(dict_size))
    else:
        raise ValueError("Mode not valid. Please use either binary or "
                         "ascii")

    # Temporary variables needed for grouping substrings in the string
    # to be compressed:
    #   c_val: every single character in the string
    #   w_val: current substring
    #   wc_val: w_val + c_val

    w_val: str = ""
    result: list = []
    for c_val in data_string:
        wc_val = w_val + c_val

        if wc_val in dictionary:
            w_val = wc_val
        else:
            result.append(dictionary[w_val])
            # Add wc to the dictionary.
            dictionary[wc_val] = dict_size
            dict_size += 1
            w_val = c_val

    # Output the code for w.
    if w_val:
        result.append(dictionary[w_val])
    if verbose:
        print("length of input string:", len(data_string))
        print("length of dictionary:", len(dictionary))
        print("length of result:", len(result))
    return result, len(dictionary), dictionary


def compute_description_length(data_array: np.ndarray, classic: bool = False):
    """
    Computes description lenght l_{LZW} as described in TN000344.
    """
    entry: str = "".join([np.str(np.int(x)) for x in data_array])
    compressedstring: list
    len_dict: int
    compressedstring, len_dict, _ = lzw_compress(entry)

    # Description length calculation
    dlength: float = np.log2(np.log2(max(compressedstring))) + np.log2(
        max(compressedstring)) * len(compressedstring)

    if classic:
        # old way ... more for LZ than LZW:
        dlength = len_dict * np.log2(len_dict)

    return dlength


def compute_rho0(data_array: np.ndarray):
    """
    Computes rho0 metric (bits/Sample). Ref: TN000344 Starlab
    / Luminous
    """
    rho_0: float = compute_description_length(data_array) / len(data_array)
    return rho_0


def shannon_entropy(array_labels: np.ndarray):
    """
    Computes entropy of label distribution.
    """
    array_labels: np.ndarray = np.asarray(array_labels, int)
    n_array_labels: int = len(array_labels)

    if n_array_labels <= 1:
        return 0
    counts: np.ndarray = np.bincount(array_labels)
    probs: np.ndarray = counts * 1.0 / n_array_labels * 1.0
    n_classes: int = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    # Compute standard entropy.
    ent: float = 0.
    for i in probs:
        ent -= i * np.log2(i)

    return ent


def compute_rho1(data_array: np.ndarray):
    """
    Computes rho1 metric (bits/Sample). Ref: TN000344 /
    Starlab Luminous
    """
    _dlength: float = compute_description_length(data_array)
    rho1: float = _dlength / shannon_entropy(data_array) / len(data_array)
    return rho1


def compute_rho2(data_array: np.ndarray):
    """
    Computes rho2 metric (bits/Sample). Ref: TN00044.
    """
    rho2: float = compute_rho0(data_array) - shannon_entropy(data_array)
    return rho2
