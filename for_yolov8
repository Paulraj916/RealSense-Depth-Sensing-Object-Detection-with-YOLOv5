import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Set the depth scale
depth_scale = 0.0010000000474974513

# Create spatial and temporal filters for depth image
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# Main loop
try:
    while True:
        # Get the latest frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert the frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply filters to depth image
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the depth image to meters
        depth_image = depth_image * depth_scale

        # Detect objects using YOLOv8
        results = model(color_image)

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()

                if confidence < 0.5:
                    continue  # Skip detections with low confidence

                # Calculate the distance to the object
                object_depth = np.median(depth_image[y1:y2, x1:x2])
                label = f"{object_depth:.2f}m"

                # Draw a rectangle around the object
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (252, 119, 30), 2)

                # Draw the bounding box
                cv2.putText(color_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

                # Print the object's class and distance
                print(f"{model.names[int(class_id)]}: {object_depth:.2f}m")

        # Show the image
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)
finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
