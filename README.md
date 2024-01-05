# BURST: BUbble Rise and Size Tracking

![output_20230308_Test4_WideFOV_Macro_v3_npx2_unsharp_cap__LR_video_00_00_06_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/87b686a4-1b76-4520-806c-662f241b3298)


# DESCRIPTION
BURST is a methodology to measure the sizes and rise velocities of underwater bubbles (gas or oil) from dual-camera footage and to estimate the relates volumetric flow rates. The BURST python script (BURST.py) batches all steps of the methodology, including the automated detection and matching of bubbles from the camera footage, and outputs the bubble size distribution (BSD), the bubble rise speed (BRS) and bubble flow rate.
Bubble detection uses the yolo neural network (currently Yolo-v4, but the code can be updated to use a newer version). Yolo weights must be computed separately by training the neural network as described here (https://github.com/AlexeyAB/darknet).
The bubble plume to quantify must be imaged with a dual-camera platform, with the two cameras ideally positioned at a ralative angle close to 90°. Both video files must be synchronised beforehand (the video files may have different frame rates but the first frame of each video file must be perfectly synchronized). 
The main script's inputs are the two video files (synchronized), the distortion parameters or each camera, a selection of synchronized camera frames from the two video files showing a common reference target with known dimensions (e.g. checkerboard pattern) for the camera position estimation, and the YOLO weight files created during the neural network training. All script's inputs are defined in a configuration file.


![cap_B2_IS327_LR_video_00_00_11_02](https://github.com/BUbbleRST/BURST.github/assets/16003542/468b2f3b-5f50-436a-8ede-943ea672c222)
BURST methodology applied to video footage capturing a natural methane emission on the seafloor at water depths exceeding 300 meters. In the left-hand video, the white band at the top indicates a masked-out region during bubble detection, eliminating areas outside the field of view of the other camera for enhanced robustness in left/right bubble matching.


![cap_B3_IS341_LR_video_00_00_15_01](https://github.com/BUbbleRST/BURST.github/assets/16003542/1419499a-a2ed-4c7c-876d-aba8c6a689e4)
BURST methodology applied to video footage capturing multiple natural methane emissions on the seafloor at water depths exceeding 300 meters. Successful bubble detection is achieved despite suboptimal imaging conditions and varying footage quality. Note the masked regions in both images, used to exclude bubble plumes falling outside the field of view of the other camera, enhancing the accuracy of the analysis.

# TEST IT YOURSELF
Test videos with corresponding camera parameters, YOLO weights, etc. are provided with the script. To test BURST, follow these steps:
1. Download all the files and subdirectories
2. Unpack the YOLO weight from the tar archive located inside the "dnn_model_bubbles" subdirectory. The YOLO weight file is required for the bubble detection (it is tar-archived only because the file size is otherwise too large for GitHub)
3. Install all python modules required to run BURST. A non-exhaustive list of the required python modules includes: configparser, numpy, scipy, opencv-python, matplotlib, scikit-image, filterpy, lap.
4. Adjust the file paths in the BURST configuration file (BURST_config.txt) according to your operating system (the provided configuration file uses linux path formats).
5. Run the BURST.py script.
   
