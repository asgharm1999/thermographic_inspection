from pathlib import Path
from rosbags.highlevel import AnyReader
from sensor_msgs.msg import Image
import numpy as np
from skimage import exposure, img_as_ubyte
import matplotlib.pyplot as plt
import cv2
from PCT import thermographic_preprocessing, PCT, SPCT, ESPCT
from PPT import PPT
from display import display


# Path to your .db3 ROS2 bag file
bag_path = '/home/arilab-pc/Downloads/thermography_02_28_2024/blackAlRectPlate30s.bag/'
bag_posix_path = Path(bag_path)

# Topic you are interested in
topic_name = '/image_raw'

# Initialize overall max and min values
overall_max_value = None
overall_min_value = None
frames = None

# Create reader instance to display the first frame and select RoI
with AnyReader([bag_posix_path]) as reader:
    connections = [x for x in reader.connections if x.topic == topic_name]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        image_data = np.frombuffer(msg.data, dtype=np.uint16)

        # Normalize and scale the intensity using custom min and max values
        image_data = image_data.reshape(msg.height, msg.width)
        img_8bit = exposure.rescale_intensity(image_data)
        img_8bit = img_as_ubyte(img_8bit)

        # Create window to display frame and select RoI. If user presses 's', skip to next frame.
        cv2.imshow("Press 's' to skip frame, 'c' to select RoI", img_8bit)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            cv2.destroyAllWindows()
            continue
        elif key == ord("c"):
            break

    # Select RoI
    x, y, w, h = cv2.selectROI("Select RoI", img_8bit)

cv2.destroyAllWindows()

# Create reader instance to calculate overall max and min values in high dynamic range of thermal video pixel values
with AnyReader([bag_posix_path]) as reader:
    connections = [x for x in reader.connections if x.topic == topic_name]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        image_data = np.frombuffer(msg.data, dtype=np.uint16)
        cropped_image_data = image_data.reshape(msg.height, msg.width)[y : y + h, x : x + w]
        cropped_image_data = cropped_image_data.flatten()

        if image_data is not None:
            frame_max_value = np.max(cropped_image_data)
            frame_min_value = np.min(cropped_image_data)

            # Update overall max and min values
            overall_max_value = frame_max_value if overall_max_value is None else max(overall_max_value, frame_max_value)
            overall_min_value = frame_min_value if overall_min_value is None else min(overall_min_value, frame_min_value)


# Create reader instance to display the selected RoI region in the normalized video
with AnyReader([bag_posix_path]) as reader:
    connections = [x for x in reader.connections if x.topic == topic_name]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        image_data = np.frombuffer(msg.data, dtype=np.uint16)

        if image_data is not None:
            # Normalize and scale the intensity using custom min and max values
            cropped_image_data = image_data.reshape(msg.height, msg.width)[y : y + h, x : x + w]
            img_8bit = exposure.rescale_intensity(cropped_image_data, in_range=(overall_min_value, overall_max_value), out_range=(0, 1))
            img_8bit = img_as_ubyte(img_8bit)

            # Display the image using OpenCV
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video", 640, 480)  # Set the window size to 800x600
            cv2.imshow("Video", img_8bit)
            cv2.waitKey(1)  # Wait for 1ms to allow the frame to be displayed
            # Stack the frames into a 3D array
            if frames is None:
                frames = cropped_image_data[np.newaxis, :]
            else:
                frames = np.concatenate((frames, cropped_image_data[np.newaxis, :]), axis=0)


print(f"Overall Maximum Value: {overall_max_value}")
print(f"Overall Minimum Value: {overall_min_value}")
print(f"Dimensions of the video: {frames.shape}")

# hot1, hot2 = thermographic_preprocessing(frames[2:, :, :], frames[0:2, :, :])

# Perform PCT
print("Performing PCT...")
numEOFs = 6
EOFs = PCT(frames, norm_method="mean reduction", EOFs=numEOFs)

# Display EOFs
print("Displaying EOFs...")
res = display(EOFs, [f"EOF{i}" for i in range(numEOFs)], bag_path)

# Save EOFs
print("Saving EOFs...")
for i, EOF in enumerate(EOFs):
    np.save(bag_path + f"EOF{i}.npy", EOF)