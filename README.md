# BURST.github
BUbble Rise and Size Tracking

# DESCRIPTION
BURST is a methodology to measure the sizes and rise velocities of underwater bubbles (gas or oil) from dual-camera footage and to estimate the relates volumetric flow rates. The BURST python script batches all steps of the methodology, including the automated detection and matching of bubbles from the camera footage, and outputs the bubble size distribution (BSD), the bubble rise speed (BRS) and bubble flow rate.
Bubble detection uses the yolo neural network (currently Yolo-v4, but the code can be updated to use a newer version). Yolo weights must be computed separately by training the neural network as described here (https://github.com/AlexeyAB/darknet).
The bubble plume to quantify must be imaged with a dual-camera platform, with the two cameras ideally positioned at a ralative angle close to 90°. Both video files must be synchronised beforehand (the video files may have different frame rates but the first frame of each video file must be perfectly synchronized). 
The main script's inputs are the two video files (synchronized), the distortion parameters or each camera, a selection of synchronized camera frames from the two video files showing a common reference target with known dimensions (e.g. checkerboard pattern) for the camera position estimation, and the YOLO weight files created during the neural network training. All script's inputs are defined in a configuration file.

