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
import argparse
from pathlib import Path
from matplotlib.animation import FuncAnimation
from PulseThermography import PulseThermography

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add the arguments
    parser.add_argument('bag_path', type=str, help='The path to the .db3 ROS2 bag file')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use args.bag_path as your bag_path
    bag_path = args.bag_path # Path to your .db3 ROS2 bag file

    if not bag_path.endswith("/"):
        bag_path += "/"

    # Check for a path to the folder if it exists. If not, create the folder and write a message to the terminal that the folder was created.
    for i in range(3):
        derivative_path = f"{bag_path}derivative_{i}/"
        if not Path(derivative_path).exists():
            print(f"Path {derivative_path} does not exist. Creating the folder...")
            Path(derivative_path).mkdir(parents=True, exist_ok=True)
            print(f"Folder {derivative_path} has been created.")
        

    #bag_path = '/home/arilab-pc/Downloads/thermography_02_28_2024/blackAlRectPlate30s.bag/'
    bag_posix_path = Path(bag_path)

    # Topic you are interested in
    topic_name = '/image_raw'

    # Initialize values for some key variables in the code
    overall_max_value = None
    overall_min_value = None
    frames = None
    raw_scaled_frames = None
    raw_message_height = 0
    raw_message_width = 0

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

    raw_message_height = msg.height
    raw_message_width = msg.width

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

    frame_number = 0  # Initialize frame number outside the loop

    # Create reader instance to display the selected RoI region in the normalized video
    with AnyReader([bag_posix_path]) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            image_data = np.frombuffer(msg.data, dtype=np.uint16)

            if image_data is not None:
                # Normalize and scale the intensity using custom min and max values
                img_8bit = exposure.rescale_intensity(image_data.reshape(msg.height, msg.width), in_range=(overall_min_value, overall_max_value), out_range=(0, 1))
                img_8bit = img_as_ubyte(img_8bit)

                frame_number += 1  # Increment frame number

                # Stack the frames into a 3D array
                if raw_scaled_frames is None:
                    raw_scaled_frames = img_8bit[np.newaxis, :]
                else:
                    raw_scaled_frames = np.concatenate((raw_scaled_frames, img_8bit[np.newaxis, :]), axis=0)

    # Save raw_scaled_frames as mp4 video
    output_path = str(bag_path + 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (raw_message_width, raw_message_height), False)
    
    for frame in raw_scaled_frames:
        # Reshape frame to match expected dimensions and convert to uint8
        frame = frame.reshape(raw_message_height, raw_message_width)  # Reshape to 480x640
        out.write(frame)
    
    out.release()
    print('Raw scaled video saved!')

    frame_number = 0  # Initialize frame number outside the loop

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

                frame_number += 1  # Increment frame number


                # Display the image using OpenCV
                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Video", 640, 480)  # Set the window size to 640 x 480 pixels
                # cv2.putText(img_8bit, 'Frame: {}'.format(frame_number), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)                
                cv2.imshow("Video", img_8bit)
                cv2.waitKey(1)  # Wait for 1ms to allow the frame to be displayed
                # Stack the frames into a 3D array
                if frames is None:
                    frames = cropped_image_data[np.newaxis, :]
                else:
                    frames = np.concatenate((frames, cropped_image_data[np.newaxis, :]), axis=0)
                    if frame_number == 1500:
                        break


    print(f"Overall Maximum Value: {overall_max_value}")
    print(f"Overall Minimum Value: {overall_min_value}")
    print(f"Dimensions of the video: {frames.shape}")

    # # Create an instance of the PulseThermography class
    # frame_rate = 60  # Set the frame rate of your thermal video
    # analysis = PulseThermography(frames, frame_rate)

    # # Set TSR settings
    # tsr_order = 5
    # analysis.set_tsr(tsr_order)

    # # Set background subtraction frames
    # frames_to_average = 5
    # analysis.set_frames_to_average(frames_to_average)

    # # Set PPT settings
    # ppt_window_type = 'hamming'
    # analysis.set_ppt(ppt_window_type, [], 1)

    # # Define the desired processing sequence
    # processing_sequence = ['estimate_flash_frame', 'subtract_reference_frame', 'perform_tsr', 'perform_ppt']

    # # Perform the analysis pipeline with the specified sequence
    # analysis.perform_analysis(processing_sequence)

    # # Access the processed data
    # tsr_temperature = analysis.tsr_temperature
    # ppt_phase = analysis.ppt_phase

    # # Visualize the results
    # analysis.imshow(tsr_temperature[:, :, 0])  # Display the first frame of TSR temperature
    # analysis.imshow(ppt_phase[:, :, 0])  # Display the first frequency bin of PPT phase



    hot1, hot2, hot1_img, hot2_img = thermographic_preprocessing(frames[1750:, :, :], frames[0:2, :, :])

    frame_index = 0
    paused = False
    replay = False

    while True:
        if not paused or replay:
            cv2.imshow('Derivative1 video', hot1_img[frame_index])
            replay = False

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Press q to stop the loop
            break
        elif key == ord('p'):  # Press p to pause/unpause the video
            paused = not paused
        elif key == ord('r'):  # Press r to replay the video
            frame_index = 0
            replay = True
        elif not paused:
            frame_index += 1

        # If we've looped to the end of the sequence, replay the video
        if frame_index == len(hot1_img):
            frame_index = 0

    cv2.destroyAllWindows()

    frame_index = 0
    paused = False
    replay = False

    while True:
        if not paused or replay:
            cv2.imshow('Derivative2 video', hot2_img[frame_index])
            replay = False

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Press q to stop the loop
            break
        elif key == ord('p'):  # Press p to pause/unpause the video
            paused = not paused
        elif key == ord('r'):  # Press r to replay the video
            frame_index = 0
            replay = True
        elif not paused:
            frame_index += 1

        # If we've looped to the end of the sequence, replay the video
        if frame_index == len(hot2_img):
            frame_index = 0

    cv2.destroyAllWindows()

    # Perform PCT
    print("Performing PCT...")
    numEOFs = 6
    
    which_deriv = [0]  # or any other list of integers

    for deriv in which_deriv:
        if deriv == 0:
            EOFs = PCT(frames[2400:, :, :], norm_method="mean reduction", EOFs=numEOFs)
            process_and_save_EOFs(bag_path, deriv, EOFs, numEOFs)
        elif deriv == 1:
            EOFs = PCT(hot1, norm_method="mean reduction", EOFs=numEOFs)
            process_and_save_EOFs(bag_path, deriv, EOFs, numEOFs)
        elif deriv == 2:
            EOFs = PCT(hot2, norm_method="mean reduction", EOFs=numEOFs)
            process_and_save_EOFs(bag_path, deriv, EOFs, numEOFs)
        else:
            print(f"Invalid value for which_deriv: {deriv}. Please enter 0, 1, or 2.")

def visualize_video(video, title="Video"):
    fig, ax = plt.subplots()
    # Normalize the 16-bit image data to [0, 1] for visualization
    norm_video = (video - np.min(video)) / (np.max(video) - np.min(video))
    im = ax.imshow(norm_video[0], cmap='hot', interpolation='nearest')
    plt.title(title)

    def update(frame):
        im.set_data(norm_video[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=range(len(video)), blit=True)
    plt.show()

def process_and_save_EOFs(bag_path, which_deriv, EOFs, numEOFs):
    filepath = f"{bag_path}derivative_{which_deriv}/"
    
    # Display EOFs
    print("Displaying EOFs...")
    res = display(EOFs, [f"EOF{i}" for i in range(numEOFs)], filepath)

    # Save EOFs
    print("Saving EOFs...")
    for i, EOF in enumerate(EOFs):
        np.save(filepath + f"EOF{i}.npy", EOF)

if __name__ == "__main__":
    main()