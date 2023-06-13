"""
Python snippets and functions to help facilitate saving and loading volumetric and surface data to be used with NMF.
"""
import os #for checking if filepaths are reasonable, and files/directories exist
if (os.name == 'nt'):
    from asyncio.windows_events import NULL
import sys #check python version, exit upon sanity check failures
import numpy as np #for doing statistics and matrix manipulation
import hdf5storage #for saving as hdf5 mat file (v7.3 on matlab); has much better compression than scipy savemat
import time #for calculating remaining time
import datetime #for calculating remaining time

#default value for script name to be used for printing to console
script_name=os.path.basename(__file__)
module_name=os.path.basename(__file__)

##### OS and System Related Functions
def check_python_version(major=3, minor=6):
    """
    using sys module, check if currently used python version is above the user provided version (3.6 default)
    """
    if sys.version_info[0] < major or sys.version_info[1] < minor:
        print("ERROR: This script requires python version %i.%i. You are using version %i.%i. Exiting." % (major, minor, sys.version_info[0], sys.version_info[1]), flush = True)
        sys.exit(1)

def exit_if_not_exist_dir(dir_path, script_name = script_name):
    """
    Using os module to check whether a directory exists, and exit using sys module if it does not exist.
    """
    if not os.path.isdir(dir_path):
        print("%s: ERROR - directory ( %s ) does not exist. Exiting." % (script_name, dir_path), flush=True)
        sys.exit(1)

def exit_if_not_exist_file(file_path, script_name = script_name):
    """
    Using os module to check whether a file exists, and exit using sys module if it does not exist.
    """
    if not os.path.isfile(file_path):
        print("%s: ERROR - file ( %s ) does not exist. Exiting." % (script_name, file_path), flush=True)
        sys.exit(1)

def exit_if_exist_dir(dir_path, script_name = script_name):
    """
    Using os module to check whether a directory exists, and exit using sys module if it does exist.
    """
    if os.path.isdir(dir_path):
        print("%s: ERROR - directory ( %s ) already exists. Exiting." % (script_name, dir_path), flush=True)
        sys.exit(1)

def exit_if_exist_file(file_path, script_name = script_name):
    """
    Using os module to check whether a file exists, and exit using sys module if it does exist.
    """
    if os.path.isfile(file_path):
        print("%s: ERROR - file ( %s ) already exists. Exiting." % (script_name, file_path), flush=True)
        sys.exit(1)

def print_flush(string_variable, script_name=script_name ):
    if isinstance(string_variable, str):
        print("%s: %s" % (script_name, string_variable), flush=True)
    else:
        print("%s: " % (script_name))
        print(string_variable, flush=True)

def print_verbose(string_variable, VERBOSE_FLAG = False, script_name=script_name ):
    if VERBOSE_FLAG:
        if isinstance(string_variable, str):
            print("%s: VERBOSE - %s" % (script_name, string_variable), flush=True)
        else:
            print("%s: VERBOSE" % (script_name))
            print(string_variable, flush=True)

def print_debug(string_variable, DEBUG_FLAG = False, script_name=script_name ):
    if DEBUG_FLAG:
        if isinstance(string_variable, str):
            print("%s: DEBUG - %s" % (script_name, string_variable), flush=True)
        else:
            print("%s: DEBUG" % (script_name))
            print(string_variable, flush=True)

def load_hdf5storage_data(
    file_path,           #type: str
    VERBOSE = False,     #type: bool
    variable_name = "X",
    script_name = script_name
):
    """
    read in hdf5storage (matlab compatible version 7.3) mat file
    
    inputs:
        file_path: full path to mat file
        VERBOSE: print additional lines to help debugging and to track progress of the function
        variable_name: variable name to extract from file_path

    outputs:
        X: data matrix of size m by n
    """

    #check if list file exists
    if not os.path.isfile(file_path):
        print("%s: %s: ERROR: mat file ( %s ) does not exist. Exiting." % (script_name, module_name, file_path), flush=True)
        sys.exit(1)
    
    #load data
    if VERBOSE:
        print("%s: %s: loading mat file ( %s )." % (script_name, module_name, file_path), flush=True)
    X = hdf5storage.loadmat(file_name = file_path, variable_names=[variable_name])[variable_name]
    return X

def calculate_error(X, W, H, SQUARED = False):
    #2022-09-27: changed from frobenius norm of (X-WH) to frobenius norm squared
    error = np.linalg.norm(X - np.matmul(W, H), ord='fro')
    if SQUARED:
        error = np.power(error, 2)
    return error

def calculate_sparsity(W):
    D = np.shape(W)[0]
    numerator = np.sum( np.abs(W), axis=0)
    denominator = np.sqrt(np.sum(np.power(W,2),axis=0))
    subtract_by = np.divide(numerator, denominator)
    del numerator, denominator

    subtract_from = np.sqrt(D)
    subtracted = subtract_from - subtract_by
    del subtract_from, subtract_by

    numerator = np.mean(subtracted)
    del subtracted

    denominator = np.sqrt(D)-1
    sparsity = np.divide( numerator, denominator)
    return sparsity

def calculate_relative_value(ref_value, new_value):
    difference = new_value - ref_value
    divided_by = np.divide(difference, ref_value)
    percentage = np.multiply(divided_by, 100)
    return percentage

def clustering_adjustedRand_fast(u, v):
    """
    python port of Aris's clustering_adjustedRand_fast.m
    """
    m = np.maximum(np.max(u, axis = 0), np.max(v, axis=0))

    va = np.zeros(shape = (m, ) )
    vb = np.zeros(shape = (m, ) )
    mat = np.zeros(shape = (m, m))

    print_debug("clustering_adjustedRand_fast: m (%i)" % (m))
    print_debug("clustering_adjustedRand_fast: va.shape, vb.shape, mat.shape")
    print_debug(va.shape)
    print_debug(vb.shape)
    print_debug(mat.shape)

    for i in np.arange(m):
        va[i] = np.sum(u == i)
        vb[i] = np.sum(v == i)
        hu = (u == i)
        for j in np.arange(m):
            hv = ( v == j)
            mat[i, j] = np.sum( np.multiply(hu, hv))

    ra = np.divide( np.sum( np.sum( np.multiply( mat, np.transpose(mat - np.ones(shape = (m,m))) ), axis = 0), axis = 0), 2.0)

    rb = np.divide( np.matmul(va, np.transpose(va - np.ones(shape = (1, m))) ), 2.0)
    
    rc = np.divide( np.matmul(vb, np.transpose(vb - np.ones(shape = (1, m))) ), 2.0)

    rn = ( np.shape(u)[0] * (np.shape(u)[0] - 1) ) / 2.0

    r = (ra-rb*rc/rn) /( 0.5*rb+0.5*rc-rb*rc/rn )
    return r

def get_binarized_mask(mask, mask_threshold = 0.8):
    """
    given a 3D data matrix, create a binary mask where the voxels of data matrix that are greater than the mask_threshold is set to 1 and rest are set to 0.

    inputs:
        mask: 3D data matrix
        mask_threshold = (0.8 by default)
    """
    binarized_mask = np.zeros(shape = mask.shape)
    binarized_mask[mask > mask_threshold] = 1.0
    return binarized_mask

def get_union_mask(mask1, mask2, mask1_threshold = 0.8, mask2_threshold = 0.8, VERBOSE = False):
    """
    take two 3D data matrices of same size, and create a union mask where either of teh two matrices are greater than some threshold value. Note that this is a binary mask with values {0, 1}

    inputs:
        mask1: 3D data matrix
        mask2: 3D data matrix
        mask1_threshold = (0.8 by default)
        mask2_threshold = (0.8 by default)
    """
    #check if two are same shape
    if mask1.shape != mask2.shape:
        print("%s: shape of mask1 is not the same as mask2" % (script_name), flush=True)
        print("%s: mask1.shape = " % (script_name))
        print(mask1.shape)
        print("%s: mask2.shape = " % (script_name))
        print(mask2.shape)
        print("%s: Exiting function.", flush=True)
        return
    
    mask1_binarized = get_binarized_mask(mask1, mask_threshold = mask1_threshold)
    if VERBOSE:
        print("%s: VERBOSE: number of voxels in mask1 > 0:" % (script_name))
        print(np.sum(mask1_binarized > 0), flush=True)

    mask2_binarized = get_binarized_mask(mask2, mask_threshold = mask2_threshold)
    if VERBOSE:
        print("%s: VERBOSE: number of voxels in mask2 > 0:" % (script_name))
        print(np.sum(mask2_binarized > 0), flush=True)

    union_mask_binarized = np.zeros(shape = mask1.shape)
    union_mask_binarized[mask1_binarized > 0] = 1.0
    union_mask_binarized[mask2_binarized > 0] = 1.0
    return union_mask_binarized

def isequal_header(nii1, nii2, VERBOSE = False):
    """
    compare q and s forms from two nii structures. If they are equal, return true, and return false if they are different.
    
    inputs:
        nii1: nibabel nifti structure 1
        nii2: nibabel nifti structure 2
    """
    header1 = nii1.header
    header2 = nii2.header

    qform1 = header1.get_qform()
    qform2 = header2.get_qform()

    sform1 = header1.get_sform()
    sform2 = header2.get_sform()

    shape1 = header1.get_data_shape()
    shape2 = header2.get_data_shape()

    voxel_dim1 = header1.get_zooms()
    voxel_dim2 = header2.get_zooms()
    
    iequal_header_boolean = True
    if not np.array_equal(qform1, qform2):
        if VERBOSE:
            print("%s: isequal_header: qform is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(sform1, sform2):
        if VERBOSE:
            print("%s: isequal_header: sform is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(shape1, shape2):
        if VERBOSE:
            print("%s: isequal_header: image shape is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(voxel_dim1, voxel_dim2):
        if VERBOSE:
            print("%s: isequal_header: voxel dimension is different." % (script_name), flush=True)
        iequal_header_boolean = False

    return iequal_header_boolean

def save_intermediate_output(output_path, mdict, OVERWRITE = False, VERBOSE = False):
    """
    save W to some output path
    """

    #if output does not exist, save W
    if not OVERWRITE:
        if os.path.isfile(output_path):
            print("%s: output_path ( %s ) already exists." % (script_name, output_path), flush=True)
            return

    if VERBOSE:
        print("%s: VERBOSE: saving output ( %s )." % (script_name, output_path), flush=True)
    hdf5storage.savemat(file_name = output_path, mdict = mdict)
    return

def load_intermediate_output(output_path, variable_name, VERBOSE = False):
    """
    load specific variable for intermediate output path
    """

    #if output does not exist, save W
    if not os.path.isfile(output_path):
        print("%s: output_path ( %s ) does not exist." % (script_name, output_path), flush=True)
        return

    if VERBOSE:
        print("%s: VERBOSE: loading variable ( %s ) from output ( %s )." % (script_name, variable_name, output_path), flush=True)
    data = hdf5storage.loadmat(file_name = output_path, variable_names = [variable_name])[variable_name]
    return data

def get_remaining_time_string(iteration, max_iter, elapsed_time, VERBOSE = False):
    """
    get a projected remaining time based on current number of iterations elapsed & total iterations
    """
    if VERBOSE:
        if iteration <= 0:
            projected_total_time = datetime.timedelta(seconds = 1)
        else:
            projected_total_time = datetime.timedelta( seconds = (max_iter / iteration) * elapsed_time)
        projected_total_time_days = int(projected_total_time.days)
        projected_total_time_hours = int(np.floor_divide(projected_total_time.seconds, 60 * 60))
        projected_total_time_minutes = int(np.floor_divide(projected_total_time.seconds - projected_total_time_hours * 60 * 60, 60))
        projected_total_time_seconds = int(projected_total_time.seconds - projected_total_time_hours * 60 * 60 - projected_total_time_minutes * 60)
        projected_total_time_string = "%i-%02d:%02d:%02d" % (projected_total_time_days, projected_total_time_hours, projected_total_time_minutes, projected_total_time_seconds)
        print("%s: VERBOSE: projected total time: %s" % (script_name, projected_total_time_string), flush=True)

    #using datetime.datetime is too much hassle
    # if (iteration <= 0):
    #     projected_remaining_time = 0
    # else:
    #     projected_remaining_time = ( (max_iter - iteration) / iteration) * elapsed_time #in seconds
    # projected_remaining_time_days = int(np.floor_divide(projected_remaining_time, 60 * 60 * 24))
    # projected_remaining_time_hours = int(np.floor_divide(projected_remaining_time, 60 * 60) - (projected_remaining_time_days) * 24)
    # projected_remaining_time_minutes = int(np.floor_divide(projected_remaining_time, 60) - ( (projected_remaining_time_days * 24 + projected_remaining_time_hours) * 60 ))
    # projected_remaining_time_seconds = int(projected_remaining_time - ( (projected_remaining_time_days * 24 * 60 + projected_remaining_time_hours * 60 + projected_remaining_time_minutes) * 60))
    
    #using datetime.timedelta
    if (iteration <= 0):
        projected_remaining_time = datetime.timedelta(seconds = 1)
    else:
        projected_remaining_time = datetime.timedelta(seconds =(( (max_iter - iteration) / iteration) * elapsed_time ) )

    projected_remaining_time_days = int(projected_remaining_time.days)
    projected_remaining_time_hours = int(np.floor_divide(projected_remaining_time.seconds, 60 * 60))
    projected_remaining_time_minutes = int(np.floor_divide(projected_remaining_time.seconds - projected_remaining_time_hours * 60 * 60, 60))
    projected_remaining_time_seconds = int(projected_remaining_time.seconds - projected_remaining_time_hours * 60 * 60 - projected_remaining_time_minutes * 60)
    projected_remaining_time_string = "%i-%02d:%02d:%02d" % (projected_remaining_time_days, projected_remaining_time_hours, projected_remaining_time_minutes, projected_remaining_time_seconds)
    return projected_remaining_time_string

def get_objective_function(X, W, trXtX = -1, OPNMF = False):
    """
    ||X-WH||_F^2, where ||*||_F is frobenius norm, and in our case of oPNMF/pNMF, H = W^TX
    Hence, ||X-WW^TX||_F^2

    We note that for opNMF with H=W'X and W'W = I, the frobenius norm |X-WW'tX|_F^2 = |X|_F^2 - |W'X|_F^2
    We can calculate |X|_F^2 prior to for loop and reuse the value instead of calculating it every update

    For pNMF iwht H=W'X, we don't have orthogonality constraint W'W=I, so the simplification is not as far and one can only simplify frobenius norm above to the following:

    """
    FULL = False
    if trXtX == -1:
        FULL = True
        OPNMF = False

    if FULL == True:
        obj = np.power(np.linalg.norm( X - np.matmul(W, np.matmul(W.T, X)) , 'fro') , 2)
    else:
        WtX = np.matmul(W.T, X)

        if OPNMF:
            obj = trXtX - np.power(np.linalg.norm(WtX, 'fro'), 2)
        else:
            obj = trXtX - 2 * np.power(np.linalg.norm(WtX, 'fro'), 2) + np.power(np.linalg.norm(np.matmul(W,WtX), 'fro'), 2)
    return obj