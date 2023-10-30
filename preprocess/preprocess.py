import numpy as np
import cv2
from PCT import PCT


def normalize(x: np.ndarray):
    """
    Normalizes a 2D array to values between 0 and 255

    Parameters
    ----------
    x : np.ndarray
        2D array to normalize

    Returns
    -------
    res : np.ndarray
        Normalized 2D array
    """
    res = np.zeros(x.shape)
    cv2.normalize(x, res, 0, 255, cv2.NORM_MINMAX)
    return res.astype("uint8")


def readVideo(path: str):
    """
    Reads a video file and converts it to a numpy array

    Parameters
    ----------
    path : str
        Path to the video file

    Returns
    -------
    video : numpy array
        3D Numpy array of the video file (frames, height, width)
    """
    # Read the video file
    vid = cv2.VideoCapture(path)
    res, ret = [], True

    while ret:
        ret, img = vid.read()
        if ret:
            res.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    vid.release()

    return np.stack(res, axis=0)


def createMask(path1: str, path2: str):
    """
    Takes in two paths to videos, and creates masked videos of the same size.
    Assumes RoI is the same for both videos.

    Parameters
    ----------
    path1 : str
        Path to first video
    path2 : str
        Path to second video

    Returns
    -------
    path1Masked : str
        Path to first masked video
    path2Masked : str
        Path to second masked video
    """
    # Define variables
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    newPath1 = path1.replace(".mp4", "_masked.mp4")
    newPath2 = path2.replace(".mp4", "_masked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    # Read videos, and skip frames that are blank
    print("Reading videos...")
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            raise Exception("Cannot read video 1")

        if np.mean(frame1) < 1:  # type: ignore
            continue
        else:
            break

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            raise Exception("Cannot read video 2")

        if np.mean(frame2) < 1:  # type: ignore
            continue
        else:
            break

    # Create window to display frame and select ROI
    cv2.namedWindow("Select ROI")
    x, y, w, h = cv2.selectROI("Select ROI", frame1)

    # Create masked videos
    writer1 = cv2.VideoWriter(newPath1, fourcc, fps, (w, h))
    writer2 = cv2.VideoWriter(newPath2, fourcc, fps, (w, h))

    # Get ROI for each frame
    print("Creating masked videos...")

    roiFrame = frame1[y : y + h, x : x + w]
    writer1.write(roiFrame)
    while True:
        ret, frame = cap1.read()
        if not ret:
            break

        roiFrame = frame[y : y + h, x : x + w]
        writer1.write(roiFrame)

    roiFrame = frame2[y : y + h, x : x + w]
    writer2.write(roiFrame)
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        roiFrame = frame[y : y + h, x : x + w]
        writer2.write(roiFrame)

    # Release resources
    cap1.release()
    cap2.release()
    writer1.release()
    writer2.release()
    cv2.destroyAllWindows()

    return newPath1, newPath2


def preprocess(coldPath: str, hotPath: str, savePath: str):
    """
    Preprocesses videos for PCT

    Parameters
    ----------
    coldPath : str
        Path to cold video
    hotPath : str
        Path to hot video
    savePath : str
        Path to save EOFs
    """
    # Create masked videos
    print("Creating mask...")
    coldMask, hotMask = createMask(coldPath, hotPath)

    # Read masked videos
    print("Reading masked videos...")
    cold, hot = readVideo(coldMask), readVideo(hotMask)

    # Perform PCT
    print("Performing PCT...")
    EOF1, EOF2 = PCT(hot)

    # Save EOFs
    print("Saving EOFs...")
    EOF1, EOF2 = normalize(EOF1), normalize(EOF2)
    EOF1 = cv2.applyColorMap(EOF1, cv2.COLORMAP_JET)
    EOF2 = cv2.applyColorMap(EOF2, cv2.COLORMAP_JET)

    cv2.imshow("EOF1", EOF1)
    cv2.imshow("EOF2", EOF2)

    cv2.imwrite(savePath + "_EOF1.png", EOF1)
    cv2.imwrite(savePath + "_EOF2.png", EOF2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    preprocess(
        "videos/2023-10-30-10-before-left-angled.mp4",
        "videos/2023-10-30-10-after-left-angled.mp4",
        "images/2023-10-30-10-left-angled",
    )
