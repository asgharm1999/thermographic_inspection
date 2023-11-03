'''
PCT.py

Provides function to perform Principal Component Thermography (PCT) on a video
'''


import numpy as np


def PCT(video: np.ndarray, norm_method: str = "col-wise standardize", EOFs: int = 2):
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
        res.append(U[:,i].reshape(h, w))
    
    return res
