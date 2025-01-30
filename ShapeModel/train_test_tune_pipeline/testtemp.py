import cv2, os
from ultralytics import YOLO
import matplotlib as plt
import numpy as np

# Load the YOLO model
model_dir = "ShapeModel/train_test_tune_pipeline/models"
model_name = "yolo11m.pt"
model = YOLO(os.path.join(model_dir,model_name))

# Open the video file
data_path = "ShapeModel/train_test_tune_pipeline/data/videos"
video = "DJI_0029.MP4"
video_path = os.path.join(data_path, video)
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = os.path.join(data_path, video[:-4]+"_annotated.mp4") 

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.save_txt 
        print(f"SPEED: ",results[0].speed)
        print(f"SUMMARY: ", results[0].summary())
        #print(f"NAMES: ", results[0].names)
        print(f"PROBS: ", results[0].probs)
        results[0].save_txt("ShapeModel/train_test_tune_pipeline/test.txt", True)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame (optional)
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

