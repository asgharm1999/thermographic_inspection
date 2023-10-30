'''
PCT.py

Provides function to perform Principal Component Thermography (PCT) on a video
'''


import numpy as np


def PCT(video: np.ndarray):
    """
    Performs Principal Component Thermography (PCT) on a video

    Parameters
    ----------
    video : np.ndarray
        Video to perform PCT on. 3D numpy array of shape (frames, height, width) and dtype np.float32

    Returns
    -------
    EOF1 : np.ndarray
        First EOF of video. 2D numpy array of shape (height, width) and dtype np.float64
    EOF2 : np.ndarray
        Second EOF of video. 2D numpy array of shape (height, width) and dtype np.float64
    """
    # Reshape video to 2D array with shape (height*width, frames)
    h, w = video.shape[1:3]
    A = video.reshape(video.shape[0], -1).T

    # Perform standardization
    for i, row in enumerate(A):
        A[i] = (row - np.mean(row)) / np.std(row)
    
    # Perform SVD
    U, _, _ = np.linalg.svd(A, full_matrices=False)

    # Return first two EOFs
    return U[:, 0].reshape(h, w), U[:, 1].reshape(h, w)
