"""
Provides the PCT function for Principal Component Thermography
"""

# Import statements
import cv2
import numpy as np
from sklearn.decomposition import PCA


def readVideo(path: str, channels: int = 3):
    """
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
        3D Numpy array of the video file (frames, height, width)
    """
    # Read the video file
    vid = cv2.VideoCapture(path)
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    matrix = np.empty((frames, height, width), np.dtype("uint8"))

    frameCount = 0
    returned = True

    while frameCount < frames and returned:
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


def normalize(x):
    """
    Normalize a 2D numpy array to 0-255

    Parameters
    ----------
    x : numpy array
        Numpy array to normalize

    Returns
    -------
    x : numpy array
        Normalized numpy array
    """
    res = np.zeros(x.shape)
    cv2.normalize(x, res, 0, 255, cv2.NORM_MINMAX)
    return res.astype("uint8")


def PCT(video: np.ndarray, path: str, method: str = "PCA"):
    """
    Performs Principal Component Thermography on a video with 1 channel

    Parameters
    ----------
    video : numpy array
        Numpy array of the video file
    path : str
        Path to save results in (without extension)
    method : str
        Method to use for PCT. Can be 'PCA' or 'SVD'
    """

    # Reshape video from (frames, height, width) to (height*width, frames)
    height, width = video.shape[1], video.shape[2]

    video = video.reshape(video.shape[0], height * width)
    video = video.T

    if method == "PCA":
        # Perform PCA on the video
        print("Performing PCA...")
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(video)

        # Get EOFs
        EOF1 = principalComponents[:, 0].reshape(height, width)
        EOF2 = principalComponents[:, 1].reshape(height, width)

        # Normalize EOFs
        print("Normalizing results...")
        EOF1 = normalize(EOF1)
        EOF2 = normalize(EOF2)

        # Turn to colormap
        EOF1 = cv2.applyColorMap(EOF1, cv2.COLORMAP_JET)
        EOF2 = cv2.applyColorMap(EOF2, cv2.COLORMAP_JET)

        # Show EOFs as thermograms
        cv2.imshow("EOF1", EOF1)
        cv2.imshow("EOF2", EOF2)

        # Save EOFs as images
        path = path.removeprefix("videos/")
        year, month, day, distance, _, side = path.split("-")
        newPath = f"images/{year}-{month}-{day}-{distance}-{side}"

        cv2.imwrite(newPath + "_EOF1.png", EOF1)
        cv2.imwrite(newPath + "_EOF2.png", EOF2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif method == "SVD":
        # Standardize video
        mean = np.mean(video, axis=0)
        stdDev = np.std(video, axis=0)
        epsilon = 1e-5
        video = (video - mean) / (stdDev + epsilon)

        # Perform SVD on the video
        print("Performing SVD...")
        U, _, _ = np.linalg.svd(video, full_matrices=False)

        # Get EOFs
        EOF1 = U[:, 0].reshape(height, width)
        EOF2 = U[:, 1].reshape(height, width)

        # Normalize EOFs
        EOF1 = normalize(EOF1)
        EOF2 = normalize(EOF2)

        # Turn to colormap
        EOF1 = cv2.applyColorMap(EOF1, cv2.COLORMAP_JET)
        EOF2 = cv2.applyColorMap(EOF2, cv2.COLORMAP_JET)

        # Show EOFs as thermograms
        cv2.imshow("EOF1", EOF1)
        cv2.imshow("EOF2", EOF2)

        # Save EOFs as images
        path = path.removeprefix("videos/")
        year, month, day, distance, _, side, angle = path.split("-")
        newPath = f"images/{year}-{month}-{day}-{distance}-{side}-{angle}"

        cv2.imwrite(newPath + "_EOF1.png", EOF1)
        cv2.imwrite(newPath + "_EOF2.png", EOF2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def createMask(vid1Name, vid2Name, debug: bool = False):
    """
    Creates a mask for the videos by selecting a region of interest (ROI), cropping
     the video to that region, and saving the cropped video.

    Assumes that the videos have the same dimensions, the ROI is the same for
    both videos, and they have an .mp4 extension.

    Parameters
    ----------
    vid1Name : str
        Path to the first video (without extension)
    vid2Name : str
        Path to the second video (without extension)

    Returns
    -------
    output1Name : str
        Path to the first cropped video (with extension)
    output2Name : str
        Path to the second cropped video (with extension)
    """

    # Define variables
    output1Name = vid1Name + "_mask.mp4"
    output2Name = vid2Name + "_mask.mp4"

    cap1 = cv2.VideoCapture(vid1Name + ".mp4")
    cap2 = cv2.VideoCapture(vid2Name + ".mp4")
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Read the first frame to determine frame dimensions
    ret, frame = cap1.read()
    if not ret:
        print("Error: Cannot read video file.")
        exit()

    # First few frames may be blank, so allow frame skipping
    while np.mean(frame) < 10:
        ret, frame = cap1.read()
        if not ret:
            print("Error: Cannot read video file.")
            exit()

    # Create a window to display the video and select ROI
    cv2.namedWindow("Select ROI")
    x, y, w, h = cv2.selectROI("Select ROI", frame)

    # Create the video writer
    writer1 = cv2.VideoWriter(output1Name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer2 = cv2.VideoWriter(output2Name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Get the ROI for each frame
    print("Cropping videos...")
    roiFrame = frame[y : y + h, x : x + h]
    writer1.write(roiFrame)
    while True:
        ret, frame = cap1.read()
        if not ret:
            break

        # Crop the frame to the selected ROI
        roiFrame = frame[y : y + h, x : x + w]

        # Display the ROI frame
        if debug:
            cv2.imshow("ROI Frame 1", roiFrame)

        # Write the frame to the video file
        writer1.write(roiFrame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        if np.mean(frame) < 10:
            continue

        # Crop the frame to the selected ROI
        roiFrame = frame[y : y + h, x : x + w]

        # Display the ROI frame
        if debug:
            cv2.imshow("ROI Frame 2", roiFrame)

        # Write the frame to the video file
        writer2.write(roiFrame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release resources
    cap1.release()
    cap2.release()
    writer1.release()
    writer2.release()
    cv2.destroyAllWindows()

    return output1Name, output2Name


def preProcess(beforeFile: str, afterFile: str, debug: bool = False):
    """
    Highlights defects by performing 'cold substraction' on the heated and unheated videos

    Parameters
    ----------
    beforeFile : str
        Path to the unheated video (without extension)
    afterFile : str
        Path to the heated video (without extension)
    """
    # Create Mask
    print("Creating mask...")
    afterMask, beforeMask = createMask(afterFile, beforeFile, debug)

    # Read the videos
    afterVid = readVideo(afterMask)
    beforeVid = readVideo(beforeMask)

    print("Performing cold subtraction...")
    # Get the pixelwise mean across all frames
    afterMean = np.mean(afterVid, axis=0)
    beforeMean = np.mean(beforeVid, axis=0)

    # Normalize the means
    afterNormalized = normalize(afterMean)
    beforeNormalized = normalize(beforeMean)

    # Get the difference between the heated and unheated means
    diff = afterNormalized - beforeNormalized

    # Show the results

    copyAfter = afterNormalized.copy()
    copyBefore = beforeNormalized.copy()

    cv2.applyColorMap(copyAfter, cv2.COLORMAP_JET)
    cv2.applyColorMap(copyBefore, cv2.COLORMAP_JET)
    cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    cv2.imshow("After (Mean + Normalized)", copyAfter)
    cv2.imshow("Before (Mean + Normalized)", copyBefore)
    cv2.imshow("Difference", diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite(afterFile + "_processed.png", copyAfter)
    # cv2.imwrite(beforeFile + "_processed.png", copyBefore)
    # cv2.imwrite(afterFile + "_diff.png", diff)

    # Perform PCT on the after video if wanted
    doPCT = input("Perform PCT on the after video? (y/n): ")

    if doPCT == "y":
        PCT(afterVid, afterFile, method="SVD")

    # # Get pixelwise mean of cold video
    # print("Getting pixelwise mean of cold video...")
    # beforeMean = np.mean(beforeVid, axis=0)

    # # Normalize cold frame
    # print("Normalizing cold frame...")
    # beforeNormalized = normalize(beforeMean)
    # cv2.imshow("Before (Mean + Normalized)", beforeNormalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Normalize all frames in hot video
    # print("Normalizing hot video...")
    # for i in range(afterVid.shape[0]):
    #     afterVid[i] = normalize(afterVid[i])
    #     cv2.imshow("After (Normalized)", afterVid[i])
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Subtract cold frame from all hot frames
    # print("Subtracting cold frame from hot video...")
    # for i in range(afterVid.shape[0]):
    #     afterVid[i] = afterVid[i] - beforeNormalized

    # # Perform PCT on the hot video
    # print("Performing PCT on hot video...")
    # PCT(afterVid, afterFile, method="SVD")


# Test code
if __name__ == "__main__":
    preProcess(
        "videos/2023-10-30-10-before-left-angled",
        "videos/2023-10-30-10-after-left-angled",
        debug=False,
    )
