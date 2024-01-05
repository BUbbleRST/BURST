# BURST: BUbble Rise and Size Tracking

![output_20230308_Test4_WideFOV_Macro_v3_npx2_unsharp_cap__LR_video_00_00_06_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/87b686a4-1b76-4520-806c-662f241b3298)

# DESCRIPTION
BURST is a methodology to measure the sizes and rise velocities of underwater bubbles (gas or oil) from dual-camera footage and to estimate the relates volumetric flow rates. The BURST python script batches all steps of the methodology, including the automated detection and matching of bubbles from the camera footage, and outputs the bubble size distribution (BSD), the bubble rise speed (BRS) and bubble flow rate.
Bubble detection uses the yolo neural network (currently Yolo-v4, but the code can be updated to use a newer version). Yolo weights must be computed separately by training the neural network as described here (https://github.com/AlexeyAB/darknet).
The bubble plume to quantify must be imaged with a dual-camera platform, with the two cameras ideally positioned at a ralative angle close to 90°. Both video files must be synchronised beforehand (the video files may have different frame rates but the first frame of each video file must be perfectly synchronized). 
The main script's inputs are the two video files (synchronized), the distortion parameters or each camera, a selection of synchronized camera frames from the two video files showing a common reference target with known dimensions (e.g. checkerboard pattern) for the camera position estimation, and the YOLO weight files created during the neural network training. All script's inputs are defined in a configuration file.

![cap__LR_video_InSitu1_unsharp_00_03_13_03](https://github.com/BUbbleRST/BURST.github/assets/16003542/9b76fe0d-2c11-44e9-b113-8acfe74f4e32)

![cap__LR_video_InSitu2_20231117_00_00_11_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/550e3066-723f-4c44-bbca-9ac094709a82)
