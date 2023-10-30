import numpy as np
from scipy.fft import fft 
import cv2


def PPT(video):
    """
    Takes in a video as a 3D numpy array, and performs Pulsed Phase Infrared
    Thermography post-processing on it. Returns a 2D numpy array of the phase 
    image.

    Parameters
    ----------
    video : np.ndarray
        Video to perform PPIT on. 3D numpy array of shape (frames, height, width) and dtype np.float32
    
    Returns
    -------
    phaseImage : np.ndarray
        Phase image of the video. 2D numpy array of shape (height, width)
    """
    frames, height, width = video.shape
    video = np.moveaxis(video, 0, -1)

    # Take FFT of each pixel (temporal profile)
    fftProfile = np.zeros((height, width, frames), dtype=np.complex_)
    for i in range(height):
        for j in range(width):
            fftProfile[i,j,:] = fft(video[i,j])
    
    # Compute phase image
    phaseImage = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            phase = np.angle(fftProfile[i,j,:])
            phaseImage[i,j] = np.max(phase)

    return phaseImage