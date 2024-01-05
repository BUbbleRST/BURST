# BURST: BUbble Rise and Size Tracking

![output_20230308_Test4_WideFOV_Macro_v3_npx2_unsharp_cap__LR_video_00_00_06_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/87b686a4-1b76-4520-806c-662f241b3298)

# DESCRIPTION
BURST is a methodology to measure the sizes and rise velocities of underwater bubbles (gas or oil) from dual-camera footage and to estimate the relates volumetric flow rates. The BURST python script batches all steps of the methodology, including the automated detection and matching of bubbles from the camera footage, and outputs the bubble size distribution (BSD), the bubble rise speed (BRS) and bubble flow rate.
Bubble detection uses the yolo neural network (currently Yolo-v4, but the code can be updated to use a newer version). Yolo weights must be computed separately by training the neural network as described here (https://github.com/AlexeyAB/darknet).
The bubble plume to quantify must be imaged with a dual-camera platform, with the two cameras ideally positioned at a ralative angle close to 90°. Both video files must be synchronised beforehand (the video files may have different frame rates but the first frame of each video file must be perfectly synchronized). 
The main script's inputs are the two video files (synchronized), the distortion parameters or each camera, a selection of synchronized camera frames from the two video files showing a common reference target with known dimensions (e.g. checkerboard pattern) for the camera position estimation, and the YOLO weight files created during the neural network training. All script's inputs are defined in a configuration file.

![cap_B2_IS327_LR_video_00_00_11_02](https://github.com/BUbbleRST/BURST.github/assets/16003542/468b2f3b-5f50-436a-8ede-943ea672c222)
BURST applied to footage of natural methane emissions on the seepbed at water depths in excess of 300 m. The white band along the top of the left-hand video indicates that this part of the frame is masked out during the bubble detection. Such masks are used to exclude regions of an image that fall outside of the field of view of the other camera. This helps making the left/right bubble matching more robust.

![cap_B3_IS341_LR_video_00_00_15_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/1419499a-a2ed-4c7c-876d-aba8c6a689e4)
BURST applied to footage of natural methane emissions on the seepbed at water depths in excess of 300 m. Bubbles are successfully detected despite the suboptimal imaging conditions and footage quality. Note the masked regions in both images: they are used to are used to exclude mask bubble plumes that fall outside of the field of view of the other camera.
