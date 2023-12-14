import numpy as np
import cv2
from PCT import PCT, SPCT, ESPCT
from PPT import PPT
from display import display
import os

def readVideo(path):
    """Reads a video file and returns a numpy array of the frames.

    Args:
        path (str): Path to video file

    Returns:
        np.ndarray: Numpy array of frames
    """
    vid = cv2.VideoCapture(path)
    res, ret = [], True

    while ret:
        ret, img = vid.read()
        if ret:
            res.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    vid.release()

    return np.stack(res, axis=0)


def createMask(path1, path2):
    """Takes two videos and creates a mask for each video with the same RoI.

    Args:
        path1 (str): Path to first video
        path2 (str): Path to second video

    Raises:
        Exception: Cannot read video 1
        Exception: Cannot read video 2

    Returns:
        (str, str): Paths to masked videos
    """
    # Define variables
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    newPath1 = path1.replace(".avi", "_masked.mp4")
    newPath2 = path2.replace(".avi", "_masked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Read videos, and skip frames that are blank
    print("Reading videos...")
    while True:
        ret, frame1 = cap1.read()
        if not ret:
            raise Exception("Cannot read video 1")

        if np.mean(frame1) < 1:
            continue
        else:
            break

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            raise Exception("Cannot read video 2")

        if np.mean(frame2) < 1:
            continue
        else:
            break

    # Create window to display frame and select RoI. If user presses 's', skip
    # to next frame.
    while True:
        cv2.imshow("Press 's' to skip frame, 'c' to select RoI", frame1)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            cv2.destroyAllWindows()

            ret, frame1 = cap1.read()
            if not ret:
                raise Exception("Cannot read video 1")

            continue
        elif key == ord("c"):
            break

    # Select RoI
    x, y, w, h = cv2.selectROI("Select RoI", frame1)

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


def preprocess(coldPath, hotPath, savePath, method="PCT", options={}):
    """Preprocesses two videos and saves the results.

    Args:
        coldPath (str): Path to cold video
        hotPath (str): Path to hot video
        savePath (str): Path to save results, excluding file extension
        method (str, optional): The preprocessing method to use. Defaults to "PCT".

    Raises:
        Exception: Invalid method

    Returns:
        str: Path to figure
    """
    # Create masked videos
    print("Creating mask...")
    coldMask, hotMask = createMask(coldPath, hotPath)

    # Read masked videos
    print("Reading masked videos...")
    cold, hot = readVideo(coldMask), readVideo(hotMask)

    # Create the folder to save results
    print("Creating folder to save results...")
    try:
        os.mkdir(savePath)
    except FileExistsError:
        pass

    # print(thermographic_preprocessing(hot, cold))

    if method == "PCT":
        # Perform PCT
        print("Performing PCT...")
        numEOFs = int(options.get("numEOFs", 6))
        EOFs = PCT(hot, norm_method="mean reduction", EOFs=numEOFs)

        # Display EOFs
        print("Displaying EOFs...")
        res = display(EOFs, [f"EOF{i}" for i in range(numEOFs)], savePath)

        # Save EOFs
        print("Saving EOFs...")
        for i, EOF in enumerate(EOFs):
            np.save(savePath + f"EOF{i}.npy", EOF)

        return res

    elif method == "SPCT":
        # Perform SPCT
        print("Performing SPCT...")
        numEOFs = int(options.get("numEOFs", 6))
        EOFs = SPCT(hot, EOFs=numEOFs)

        # Display EOFs
        print("Displaying EOFs...")
        res = display(EOFs, [f"EOF{i}" for i in range(numEOFs)], savePath)

        # Save EOFs
        print("Saving EOFs...")
        for i, EOF in enumerate(EOFs):
            np.save(savePath + f"SPCT-EOF{i}.npy", EOF)

        return res

    elif method == "ESPCT":
        # Perform ESPCT
        print("Performing ESPCT...")
        numEOFs = int(options.get("numEOFs", 6))
        EOFs = ESPCT(hot, k=8, EOFs=numEOFs)

        # Display EOFs
        print("Displaying EOFs...")
        res = display(EOFs, [f"EOF{i}" for i in range(numEOFs)], savePath)

        # Save EOFs
        print("Saving EOFs...")
        for i, EOF in enumerate(EOFs):
            np.save(savePath + f"ESPCT-EOF{i}.npy", EOF)

        return res

    elif method == "PPT":
        # Perform PPIT
        print("Performing PPT...")
        phaseImage = PPT(hot)

        # Display phase image
        print("Displaying phase image...")
        res = display([phaseImage], ["Phase Image"], savePath)

        # Save phase image
        print("Saving phase image...")
        np.save(savePath + "phaseImage.npy", phaseImage)

        return res

    else:
        raise Exception("Invalid method")


if __name__ == "__main__":
    preprocess(
        "videos/2023_12_11_cold_15.avi",
        "videos/2023_12_11_hot_15.avi",
        "images/2023_12_11_small_15/",
        method="PCT",
    )
