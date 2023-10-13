# Import libraries
import cv2
import stitching

def stitchImage(imagePaths: list[str], savePath: str) -> bool:
    '''
    Stitches images together

    Parameters
    ----------
    imagePaths : list[str]
        List of image paths
    savePath : str
        Path to save the stitched image

    Returns
    -------
    status: bool
        Whether the stitching was successful
    '''
    images = [cv2.imread(path) for path in imagePaths]

    # Show images
    # for image in images:
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)

    stitcher = stitching.Stitcher()
    res = stitcher.stitch(images)

    if res is None:
        print("Error stitching images")
        return False
    
    cv2.imwrite(savePath, res)
    return True


def stitchVideo(videoPaths: list[str], savePath: str) -> bool:
    '''
    Stitches videos together

    Parameters
    ----------
    videoPaths : list[str]
        List of video paths
    savePath : str
        Path to save the stitched video

    Returns
    -------
    status: bool
        Whether the stitching was successful
    '''
    videos = [cv2.VideoCapture(path) for path in videoPaths]
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        savePath, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        videos[0].get(cv2.CAP_PROP_FPS),
        (width, height)
    )
    stitcher = cv2.Stitcher.create()

    def cleanUp():
        for video in videos:
            video.release()
        writer.release()

    while True:
        # Read a frame from each video
        frames, stop = [], False
        for video in videos:
            ret, frame = video.read()

            # If a video has ended, stop stitching
            if not ret:
                stop = True
                break
            frames.append(frame)
        
        if stop:
            break
        
        status, result = stitcher.stitch(frames)
        if status == cv2.STITCHER_OK:
            writer.write(result)
        elif status == cv2.STITCHER_ERR_NEED_MORE_IMGS:
            print("Not enough images to stitch")
            cleanUp()
            return False
        else:
            print("Error stitching images")
            cleanUp()
            return False
        
    cleanUp()
    return True

if __name__ == "__main__":
    stitchImage(["images/test_left.jpg", "images/test_right.jpg"], "images/test_stitched.png")
