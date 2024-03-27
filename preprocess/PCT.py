"""
PCT.py

Provides function to perform Principal Component Thermography (PCT) on a video
"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
import scipy.signal as signal
import itertools

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from multiprocessing import Pool

from skimage import img_as_ubyte

def apply_median_filter(video, kernel_size=3):
    # Apply a median filter frame by frame
    filtered_video = np.array([median_filter(frame, size=kernel_size) for frame in video])
    return filtered_video

def fit_polynomial(time, pixel_values):
    # Fit a 5th order polynomial and return the coefficients
    coefficients = np.polyfit(time, pixel_values, 5)
    return coefficients

def subtract_reference_frame(video, reference_frame):
    # Subtract reference frame from each frame of the video
    subtracted_video = video - reference_frame
    return subtracted_video

def normalize_video(video):
    # Normalize video to ensure all values are positive and non-zero
    min_val = np.min(video)
    # Shift video by min_val+1 if min_val is less than or equal to 0
    if min_val <= 0:
        video += (np.abs(min_val) + 1)
    return video

def fit_and_evaluate_polynomial(time, pixel_values):
    # Fit a 5th order polynomial and evaluate it at each time point
    coefficients = np.polyfit(time, pixel_values, 5)
    fitted_values = np.polyval(coefficients, time)
    return coefficients, fitted_values

def compute_derivatives(coefficients, time):
    first_derivative_coeffs = np.polyder(coefficients, 1)
    second_derivative_coeffs = np.polyder(coefficients, 2)
    
    first_derivative = np.polyval(first_derivative_coeffs, time)
    second_derivative = np.polyval(second_derivative_coeffs, time)
    
    return first_derivative, second_derivative

def process_pixel(args):
    time, pixel_values = args
    coefficients, _ = fit_and_evaluate_polynomial(time, pixel_values)
    first_derivative, second_derivative = compute_derivatives(coefficients, time)
    return first_derivative, second_derivative

def thermographic_preprocessing(hot_seq, cold_seq):
    """
    Performs thermographic preprocessing on a sequence of images

    hot_seq is 3D array of shape (frames, height, width)
    cold_seq is 3D array of shape (frames, height, width)
    deriv1_seq is 3D array of shape (frames, height, width)
    deriv2_seq is 3D array of shape (frames, height, width)

    deriv1_seq and deriv2_seq should have same height and width as hot_seq
    """
      
    # Take average of cold image sequence
    cold_avg = np.mean(cold_seq, axis=0) # Example dimension (72, 30)
    print(np.max(cold_seq))
    print(np.min(cold_seq))
    
    # Subtract averaged cold image
    hot_sub = subtract_reference_frame(hot_seq, cold_avg) # Example dimension (3780, 72, 30)
    # hot_sub = hot_seq - cold_avg # Example dimension (3780, 72, 30)
    print(np.max(hot_seq))
    print(np.min(hot_seq))

    # hot_sub = hot_seq
    print(np.max(hot_sub))
    print(np.min(hot_sub))   
    # Apply refined median filter
    hot_sub = normalize_video(hot_sub)
    hot_filt = apply_median_filter(hot_sub, kernel_size=3)
    # hot_filt = np.array([signal.medfilt(frame, 3) for frame in hot_sub])
    print(np.max(hot_filt))
    print(np.min(hot_filt))

    # Compute the logarithm of the filtered video
    log_video = np.log(hot_filt)
    
    # Prepare for polynomial fitting
    frames, height, width = log_video.shape
    time = np.arange(frames)
    
    # Fit a 5th order polynomial and evaluate for every pixel
    # Create a list of all pixel values
    pixels = [(time, log_video[:, i, j]) for i in range(height) for j in range(width)]
    
    # Create a process pool and fit polynomial for each pixel
    with Pool() as p:
        results = p.map(process_pixel, pixels)

    # Unpack results
    deriv1_seq = np.array([result[0] for result in results]).reshape((frames, height, width))
    deriv2_seq = np.array([result[1] for result in results]).reshape((frames, height, width))
    
    # # Reshape the fitted pixels to the original video shape
    # fitted = np.array(fitted_pixels).reshape((frames, height, width))


    # # # Fit 5th order polynomial along time axis of log image sequence for every pixel
    # # xdata = np.arange(len(hot_filt))
    # # fitted = []
    # # for i in range(hot_filt.shape[1]):
    # #     for j in range(hot_filt.shape[2]):
    # #         # ydata = np.log(hot_filt[:, i, j])
    # #         ydata = np.log(hot_filt[:, i, j]).astype(np.float32)
    # #         coef = np.polyfit(xdata, ydata, 5)
    # #         poly = np.polyval(coef, xdata)
    # #         fitted.append(poly)
            
    # # fitted = np.array(fitted).reshape(hot_filt.shape)
    # # Check if any values in "fitted" are NaN
    # has_nan = np.isnan(fitted)

    # # Count the number of NaN instances
    # num_nan = np.sum(has_nan)

    # print("Number of NaN instances in fitted:", num_nan)

    # # Compute 1st and 2nd derivative sequences       
    # deriv1_seq = np.gradient(fitted, axis=0)
    # deriv2_seq = np.gradient(deriv1_seq, axis=0)
    # # Check if any values in "deriv1" are NaN
    # has_nan = np.isnan(deriv1_seq)

    # # Count the number of NaN instances
    # num_nan = np.sum(has_nan)
    # print("Number of NaN instances in deriv1:", num_nan)

    # # Check if any values in "deriv1" are NaN
    # has_nan = np.isnan(deriv2_seq)

    # # Count the number of NaN instances
    # num_nan = np.sum(has_nan)
    # print("Number of NaN instances in deriv2:", num_nan)

    # Rescale the values to the range [0, 255] and convert to 8-bit unsigned integer format
    deriv1_seq_img = img_as_ubyte((deriv1_seq - np.min(deriv1_seq)) / (np.max(deriv1_seq) - np.min(deriv1_seq)))
    deriv2_seq_img = img_as_ubyte((deriv2_seq - np.min(deriv2_seq)) / (np.max(deriv2_seq) - np.min(deriv2_seq)))
    return deriv1_seq, deriv2_seq, deriv1_seq_img, deriv2_seq_img


def PCT(video, norm_method = "col-wise standardize", EOFs = 2):
    """
    Performs Principal Component Thermography (PCT) on a video

    Parameters
    ----------
    video : np.ndarray
        Video to perform PCT on. 3D numpy array of shape (frames, height, width) and dtype np.float32
    norm_method : str, optional
        Method to normalize video. Can be ["col-wise standardize", "row-wise standardize", "mean reduction"]. Defaults to "col-wise standardize"
    EOFs : int, optional
        Number of EOFs to return. Defaults to 2

    Returns
    -------
    [EOF_i] : list[np.ndarray]
        List of EOFs. Each EOF is a 2D numpy array of shape (height, width)
    """
    # Reshape video to 2D array with shape (height*width, frames)
    h, w = video.shape[1:3]
    A = video.reshape(video.shape[0], -1).T

    # Perform standardization
    if norm_method == "col-wise standardize":
        mean = np.mean(A, axis=0)
        std = np.std(A, axis=0)
        epsilon = 1e-5
        A = (A - mean) / (std + epsilon)

    elif norm_method == "row-wise standardize":
        for i, row in enumerate(A):
            A[i] = (row - np.mean(row)) / np.std(row)

    elif norm_method == "mean reduction":
        mean = np.mean(A)
        A = A - mean

    # Perform SVD
    U, _, _ = np.linalg.svd(A, full_matrices=False)

    # Return EOFs
    res = []
    for i in range(EOFs):
        res.append(U[:, i].reshape(h, w))

    return res


def SPCT(video, EOFs):
    # Format into (height*width, frames)
    h, w = video.shape[1:3]
    X = video.reshape(video.shape[0], -1).T

    # Column-wise mean reduction
    mean = np.mean(X, axis=0)
    X = X - mean

    # Perform Sparse PCA
    spca = SparsePCA(n_components=EOFs, alpha=0.5)
    res = spca.fit_transform(X)

    return [
        res[:, i].reshape(h, w) for i in range(EOFs)
    ]


def ESPCT_single_iter(X, k, max_iter=1000, tol=1e-4):
    # Normalize
    m, n = X.shape
    X = StandardScaler().fit_transform(X)

    # Initialize
    v = np.random.rand(n)  # PC vector
    v /= np.linalg.norm(v)
    p_old = np.zeros(m)  # EOF vector from previous iteration

    # Iterate
    for i in range(max_iter):
        z = X @ v  # Project X onto v, shape (m,)
        w = np.sqrt((z**2 + z**2)[:, np.newaxis])  # Compute edge weights
        index = np.argsort(w.flatten())[-k:]  # Get indices of k largest weights
        G = np.zeros((m, m))  # Group structure matrix
        G[np.unravel_index(index, (m, m))] = 1
        p = z * (G + G.T > 0)  # EOF vector, shape (m,)
        p /= np.linalg.norm(p)
        v = (X.T @ p) / np.linalg.norm(X.T @ p)  # PC vector, shape (n,)

        if np.abs(np.dot(p_old, p)) > 1 - tol:
            break
        p_old = p

    return p, v


def ESPCT(video, k, EOFs = 2):
    """
    Performs Edge-group Sparse Principal Component Thermography (ESPCT) on a video

    Parameters
    ----------
    video : np.ndarray
        Video to perform ESPCT on. 3D numpy array of shape (frames, height, width)
    k : int
        Number of neighboring pixels to consider
    EOFs : int, optional
        Number of EOFs to return. Defaults to 2

    Returns
    -------
    [EOF_i] : list[np.ndarray]
        List of EOFs. Each EOF is a 2D numpy array of shape (height, width)
    """
    res = []
    X = video.reshape(video.shape[0], -1).T

    for i in range(EOFs):
        p, v = ESPCT_single_iter(X, k)
        res.append(p.reshape(video.shape[1:3]))
        X = X - p[:, np.newaxis] @ v[np.newaxis, :]

    return res
