import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

#write your yolov5 depth scale here

depth_scale = 0.0010000000474974513

# Main loop
while True:
    
    # Get the latest frame from the camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Convert the depth image to meters
    depth_image = depth_image * depth_scale

    # Detect objects using YOLOv5
    results = model(color_image)

    # Process the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        # Calculate the distance to the object
        object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])
        label = f"{object_depth:.2f}m"

        # Draw a rectangle around the object
        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

        # Draw the bounding box
        cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

        # Print the object's class and distance
        print(f"{model.names[int(class_id)]}: {object_depth:.2f}m")

    # Show the image
    cv2.imshow("Color Image", color_image)
    cv2.waitKey(1)

# Release the VideoWriter object
out.release()
