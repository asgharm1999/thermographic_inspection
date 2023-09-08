'''
Provides the PCT function for Principal Component Thermography
'''

# Import statements
import cv2
import numpy as np
import plotly.express as px

def readVideo(path: str, channels: int = 3):
    '''
    Reads a video file and converts it to a numpy array

    Parameters
    ----------
    path : str
        Path to the video file
    channels : int
        Number of channels in the video file
    
    Returns
    -------
    video : numpy array
        Numpy array of the video file
    '''
    # Read the video file
    vid = cv2.VideoCapture(path)
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    matrix = np.empty((frames, height, width), np.dtype('uint8'))

    frameCount = 0
    returned = True

    while (frameCount < frames and returned):
        if channels == 1:
            returned, matrix[frameCount] = vid.read()
            frameCount += 1
        else:
            returned, img = vid.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            matrix[frameCount] = gray
            frameCount += 1
    
    vid.release()

    return matrix


def PCT(video: np.ndarray):
    '''
    Performs Principal Component Thermography on a video with 1 channel
    '''

    # Reshape video from (frames, height, width) to (height*width, frames)
    height, width = video.shape[1], video.shape[2]

    video = video.reshape(video.shape[0], video.shape[1]*video.shape[2])
    video = video.T

    # Perform normalization on the video
    print("Normalizing video...")
    video = video - np.mean(video, axis=0)

    # Perform SVD on the video
    print("Performing SVD...")
    U, S, V = np.linalg.svd(video, full_matrices=False)

    # Get EOFs
    U = U.T
    EOF1 = U[0].reshape(height, width)
    EOF2 = U[1].reshape(height, width)

    # Show EOFs as thermograms
    
    fig1 = px.imshow(EOF1, title='EOF1')
    fig1.write_image("EOF1.png", format="png")

    fig2 = px.imshow(EOF2, title='EOF2')
    fig2.write_image("EOF2.png")


# Test code
if __name__ == '__main__':
    print("Reading Video...")
    video = readVideo('vid-edited2.mp4')

    PCT(video)
