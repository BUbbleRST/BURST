# -*- coding: Latin-1 -*-
# BURST Bubble Rise and Size Tracking
# Version: 6.0
# Author: Yann Marcon
# Date: 12.09.2023

# Developed with: Python 3.10


# DESCRIPTION
# This script detects and matches bubbles from footage recorded with a dual-camera platform and outputs
# the bubble size distribution (BSD), the bubble rise speed (BRS) and bubble volumetric flow rate.

#
#
# INSTRUCTIONS
# Bubble detection uses the yolo neural network. Yolo weights must be computed separately.
# Both video files must be synchronised beforehand! The video files may have different frame rates but the first frame
# of each video file must be synchronised.
#
#


import sys
import os
import glob
from shutil import copyfile
import configparser
import numpy as np
import itertools
import cv2
import matplotlib
from skimage.filters import unsharp_mask

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from line_profiler_pycharm import profile

# class Bubble(object):
#     def __init__(self, bid, bbox, frameid):
#
#         self.bid = bid  # bubble id
#         self.bboxes = np.hstack((bbox, frameid), dtype=np.float)  # bounding boxes ([x, y, w, h, frameNr], one row per frame)
#
#     def add_frame(self, bbox, frameid):
#         self.bboxes = np.hstack((bbox, frameid), dtype=np.float)


class Bubbles(object):
    def __init__(self):
        self.bids = []  # bubble ids (given by the tracking)
        self.frame0 = []  # first frame where each bubble id appeared
        self.bboxes = []  # bubble bounding boxes [Left, Top, Right, Bottom, frameNr]
        # self.bboxes is a list of lists:
        # - elements of the main list corresponds to bubbles
        # - each bubble element is a list ([Left, Top, Right, Bottom, frameNr]) defining the bbox of the last detection of the bubble

    def add_frame(self, frameid, tracks):
        bubble_ids = [int(i) for i in tracks[:, -1]]  # make bubble ids integers
        for bk, bid in enumerate(bubble_ids):  # loop through bubble ids (given by the tracking), 'bk' is the index of 'bid' in the array
            if bid in self.bids:
                k = self.bids.index(bid)
                self.bboxes[k] = tracks[bk, 0:4].tolist() + [frameid]  # bounding boxes ([Left, Top, Right, Bottom, frameNr0, frameNr], only for the last detection of the bubble)
            else:
                self.bids.append(bid)
                self.frame0.append(frameid)
                self.bboxes.append(tracks[bk, 0:4].tolist() + [frameid])  # bounding boxes ([Left, Top, Right, Bottom, frameNr0, frameNr], only for the last detection of the bubble)


def round_to_int(x, d):
    if np.isscalar(x):
        x = int(round(x, d) * (10**d))
    else:
        for idx in range(x.__len__()):
            x[idx] = round(x[idx], d) * (10**d)
        x = x.astype(int)
    return x


def str_to_float(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def mouseclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global selected_corners, selected_frame
        selected_corners.append([x, y])
        cv2.drawMarker(selected_frame, selected_corners[-1], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1, line_type=cv2.LINE_AA)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def trim_image(im, rgb=[255, 255, 255]):
    bim = (im[:, :, 0] != rgb[0]) * (im[:, :, 1] != rgb[1]) * (im[:, :, 2] != rgb[2])  # binary image
    llim = np.where(np.any(bim, axis=0))[0][0]  # left limit
    rlim = np.where(np.any(bim, axis=0))[0][-1] + 1  # right limit
    tlim = np.where(np.any(bim, axis=1))[0][0]  # top limit
    blim = np.where(np.any(bim, axis=1))[0][-1] + 1  # bottom limit
    return im[tlim:blim, llim:rlim, :]


def stack_frames(frameL, frameR, direction='horizontal'):
    if direction.lower() == 'horizontal':
        if frameL.shape[0] > frameR.shape[0]:
            frameL = cv2.resize(frameL, (frameR.shape[0] * frameL.shape[1] // frameL.shape[0], frameR.shape[0]))
        elif frameL.shape[0] < frameR.shape[0]:
            frameR = cv2.resize(frameR, (frameL.shape[0] * frameR.shape[1] // frameR.shape[0], frameL.shape[0]))
        stacked_frame = np.hstack((frameL, frameR))

    elif direction.lower() == 'vertical':
        if frameL.shape[1] > frameR.shape[1]:
            frameL = cv2.resize(frameL, (frameR.shape[1], frameR.shape[1] * frameL.shape[0] // frameL.shape[1]))
        elif frameL.shape[1] < frameR.shape[1]:
            frameR = cv2.resize(frameR, (frameL.shape[1], frameL.shape[1] * frameR.shape[0] // frameR.shape[1]))
        stacked_frame = np.vstack((frameL, frameR))

    return stacked_frame


def get_chessboard_coords(chessboardSize, chessboardCellsize_mm, startpoint='BL'):
    # Chessboard 3D coordinates (in the chessboard referential)
    chessboardCoords = []
    pointmax = chessboardSize[0] * chessboardSize[1]
    if startpoint.upper()=='BL':
        for k in range(pointmax):
            x = (k % chessboardSize[0]) * chessboardCellsize_mm
            y = (k // chessboardSize[0]) * chessboardCellsize_mm
            chessboardCoords.append((x, y, 0))
    elif startpoint.upper()=='TL':
        for k in range(pointmax):
            x = (k % chessboardSize[0]) * chessboardCellsize_mm
            y = ((pointmax - (k + 1)) // chessboardSize[0]) * chessboardCellsize_mm
            chessboardCoords.append((x, y, 0))
    return np.array(chessboardCoords, np.float32)  # every array needs to be converted to np.float32 before passing to calibrateCamera because OpenCV requires floats with single precision


def get_chessboard_objpoints_and_imgpoints(calpathL, calpathR, chessboardSize, chessboardCellsize_mm, pattern='*.jpg'):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard 3D coordinates (in the chessboard referential)
    chessboardCoords = get_chessboard_coords(chessboardSize, chessboardCellsize_mm, startpoint='TL')

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    # objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    # objp *= chessboardCellsize_mm

    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d points in real world space
    imgPointsL = []  # 2d points in image plane
    imgPointsR = []  # 2d points in image plane

    imagesL = sorted(glob.glob(os.path.join(calpathL, pattern)))
    imagesR = sorted(glob.glob(os.path.join(calpathR, pattern)))

    imax = imagesL.__len__()
    for k in range(imax):
        print(f'Calibration image {k+1} / {imax}')
        imgL = cv2.imread(imagesL[k])
        imgR = cv2.imread(imagesR[k])
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR:
            objPoints.append(chessboardCoords)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPointsL.append(cornersL)
            imgPointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            # cv2.imshow(os.path.basename(image), img)
            # cv2.imshow('imgL', imgL)
            # cv2.imshow('imgR', imgR)
            # cv2.waitKey(1000)
        else:
            print(f'Image {k+1} ignored.')

    cv2.destroyAllWindows()
    # print("Object Points: ", objPoints)
    # print("Image Points: ", imgPoints)
    return objPoints, imgPointsL, imgPointsR


def select_chessboard_objpoints_and_imgpoints(calpathL, calpathR, chessboardSize, chessboardCellsize_mm, pattern='*.jpg'):
    # Chessboard 3D coordinates (in the chessboard referential)
    chessboardCoords = get_chessboard_coords(chessboardSize, chessboardCellsize_mm, startpoint='TL')
    cornersMax = chessboardSize[0] * chessboardSize[1]

    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d points in real world space
    imgPointsL = []  # 2d points in image plane
    imgPointsR = []  # 2d points in image plane

    imagesL = sorted(glob.glob(os.path.join(calpathL, pattern)))
    imagesR = sorted(glob.glob(os.path.join(calpathR, pattern)))

    imax = imagesL.__len__()
    for k in range(imax):
        print(f'Image {k+1} / {imax}')

        # LEFT IMAGE
        id = imagesL[k].rfind('.')
        ccfileL = imagesL[k][0:id] + '.txt'  # checkerboard corner file
        if os.path.exists(ccfileL):
            cornersL = np.loadtxt(ccfileL)
        else:
            cornersL = selectCheckerboardCorners(imagesL[k], cornersMax)
            if cornersL.shape[0] == cornersMax:  # case when chessboard detection works
                np.savetxt(ccfileL, cornersL, fmt='%.3f')

        # RIGHT IMAGE
        id = imagesR[k].rfind('.')
        ccfileR = imagesR[k][0:id] + '.txt'  # checkerboard corner file
        if os.path.exists(ccfileR):
            cornersR = np.loadtxt(ccfileR)
        else:
            cornersR = selectCheckerboardCorners(imagesR[k], cornersMax)
            if cornersR.shape[0] == cornersMax:  # case when chessboard detection works
                np.savetxt(ccfileR, cornersR, fmt='%.3f')

        if cornersL.__len__() > 0 and cornersL.__len__() == cornersR.__len__():
            cornersL = cornersL[:, np.newaxis, :]  # format array in the same way as cv2.findChessboardCorners
            cornersR = cornersR[:, np.newaxis, :]  # format array in the same way as cv2.findChessboardCorners
            objPoints.append(chessboardCoords)
            imgPointsL.append(cornersL.astype("float32"))
            imgPointsR.append(cornersR.astype("float32"))
        else:
            print(f'Image {k+1} ignored.')

    return objPoints, imgPointsL, imgPointsR


def selectCheckerboardCorners(frame, cornersMax):
    # IMPORTANT: select points from left to right and top to bottom of the checkerboard (not of the image!)

    global selected_frame, selected_corners
    selected_frame = cv2.imread(frame)
    selected_corners = []
    cv2.namedWindow("SelectedFrame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SelectedFrame", mouseclick)
    while selected_corners.__len__() < cornersMax:
        cv2.imshow("SelectedFrame", selected_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
    cv2.destroyAllWindows()
    corners = np.asarray(selected_corners)
    del selected_frame, selected_corners

    return corners


def find_objects(net, frame, output_layers, confT, nmsT, tracker_type='sort'):

    # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Detect objects on frame with yolo v3
    #blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    bbs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                # print(class_id)
                center_x = detection[0] * width
                center_y = detection[1] * height
                w = detection[2] * width
                h = detection[3] * height

                # Rectangle coordinates
                x = center_x - w / 2
                y = center_y - h / 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                if tracker_type == 'sort':
                    # bbs.append((x, y, x+w, y+h, float(confidence), class_id))
                    bbs.append((x, y, x + w, y + h, float(confidence)))
                elif tracker_type == 'deepsort':
                    bbs.append(([x, y, w, h], float(confidence), class_id))

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confT, nmsT)

    if bbs.__len__() > 0 and indexes.__len__() > 0:
        bbs = [bbs[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]

    return bbs, class_ids


def filter_by_class(detected_objects, class_list, classes_to_exclude):
    if detected_objects.__len__() == 0 or classes_to_exclude is None or classes_to_exclude.__len__() == 0:
        return detected_objects, class_list

    keepsel = [x for x in range(class_list.__len__()) if class_list[x] not in classes_to_exclude]
    if keepsel.__len__() > 0:
        detected_objects = [detected_objects[x] for x in keepsel]
        class_list = [class_list[x] for x in keepsel]
    else:
        detected_objects = []
        class_list = []

    return detected_objects, class_list


def filter_by_position(detected_objects, zones_to_exclude):
    if detected_objects.__len__() == 0 or zones_to_exclude is None or zones_to_exclude.__len__() == 0:
        return detected_objects

    objXY = [((x[0]+x[2])/2, (x[1]+x[3])/2) for x in detected_objects]
    objXY = np.asarray(objXY)
    delsel = np.full(detected_objects.__len__(), True)
    for zone in zones_to_exclude:
        delsel *= (zone[0] <= objXY[:, 0]) * (objXY[:, 0] <= zone[2]) * (zone[1] <= objXY[:, 1]) * (objXY[:, 1] <= zone[3])
    keepsel = (~delsel).nonzero()[0]
    if keepsel.__len__() > 0:
        detected_objects = [detected_objects[x] for x in keepsel]
    else:
        detected_objects = []

    return detected_objects


def mask_frame(im, zones_to_exclude, alpha=.4):
    if zones_to_exclude is None or zones_to_exclude.__len__() == 0:
        return im

    for zone in zones_to_exclude:
        overlay = im.copy()
        cv2.rectangle(overlay, (zone[0], zone[1]), (zone[2], zone[3]), (255, 255, 255), -1)  # A filled rectangle
        im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)  # overlay transparent rectangle over image

    return im


def filter_by_time(tracks, bubbleObj, frameNr, fps, maxtimesec):
    bids = [int(i) for i in tracks[:, -1]]  # make bubble ids integers
    existing_bids = np.asarray([i for i in bids if i in bubbleObj.bids])  # list existing bids

    # filter out tracked objects depending on time
    if existing_bids.size > 0:
        idx = [bubbleObj.bids.index(i) for i in existing_bids]  # list of indexes of existing bids
        idx_bool = (frameNr - np.asarray(bubbleObj.frame0)[idx]) / fps > maxtimesec
        if any(idx_bool):
            new_bids = np.setdiff1d(bids, bubbleObj.bids)  # yields the elements of bids that are not yet in bubbleObj.bids
            keepsel = np.concatenate((existing_bids[~idx_bool], new_bids))
            keepsel = [bids.index(i) for i in keepsel]
            keepsel.sort()
            tracks = tracks[keepsel, :]
    return tracks


def update_tracker(tracker, bbs, frame, tracker_type='sort', fontsize=1.5):
    font = cv2.FONT_HERSHEY_PLAIN
    if tracker_type == 'sort':
        if bbs.__len__() > 0:
            bbs = np.asarray(bbs)
            # tracks = tracker.update(bbs)
            _, tracks = tracker.update(bbs)  # returns actual positions, not estimated positions

            for track in tracks:
                track = track.astype(np.int32)
                x = round(track[0])
                y = round(track[1])
                w = round(track[2] - track[0])
                h = round(track[3] - track[1])
                featureID = f'{track[4]}'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, featureID, (x + w + 1, y + 30), font, fontsize, (0, 255, 255), 2)

        else:  # update tracker with an empty array
            _, tracks = tracker.update(np.empty((0, 5)))

    elif tracker_type == 'deepsort':
        if bbs.__len__() > 0:
            tracks = tracker.update_tracks(bbs,
                                           frame=frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

            for track in tracks:
                track_id = track.track_id
                # cbox = track.to_ltrb(orig=True) # left, top, right, bottom
                x, y, w, h = track.to_tlwh(orig=True).astype(np.int32)  # topleft x, topleft y, width, height
                # If orig is flagged as True and this track is associated to a detection this update round, then the
                # bounding box values returned by the method will be that associated to the original detection.
                # Otherwise, it will still return the Kalman predicted values.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, track_id, (x, y + 30), fontsize, 3, (0, 255, 255), 2)

        # else:  # pass an empty array
            # TO DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return tracker, tracks, frame


def inverse_intrinsic_matrix(K):
    Kinv = np.zeros((3, 3), dtype=np.float)  # inverse of intrinsic matrix
    Kinv[0, 0] = 1 / K[0, 0]
    Kinv[1, 1] = 1 / K[1, 1]
    Kinv[0, 2] = - K[0, 2] / K[0, 0]
    Kinv[1, 2] = - K[1, 2] / K[1, 1]
    return Kinv


def CameraRay(pts2D, intrinsic, extrinsic, mu=1, distortion=None):
    # This function computes the camera ray between the camera center and the intersection onto the infinity plane.
    # Mu is the parameter of the points along the ray (mu=0 corresponds to the camera center).
    pts2D = np.asarray(pts2D)

    P = intrinsic.dot(extrinsic)  # projection matrix

    M = P[:, 0:3]  # 3x3 submatrix from P (notation from Zisserman's book p.161-162)
    p4 = P[:, 3]  # (notation from Zisserman's book p.161-162)

    Minv = np.linalg.inv(M)

    # Step 1. Undistort.
    if distortion is not None:
        pts2D_undistorted = np.array([])
        if len(pts2D) > 0:
            pts2D_undistorted = cv2.undistortPoints(np.expand_dims(pts2D, axis=1), intrinsic, distortion, P=intrinsic)
        pts2D = np.squeeze(pts2D_undistorted, axis=1)

    # Step 2. Ray
    rays = []
    pts2D = np.asarray(pts2D)
    if pts2D.ndim > 2:
        pts2D = np.squeeze(pts2D, axis=1)
    if pts2D.ndim == 1:
        pts2D = np.expand_dims(pts2D, axis=0)

    for idx in range(pts2D.shape[0]):
        # info here: https://www.geeksforgeeks.org/shortest-distance-between-two-lines-in-3d-space-class-12-maths/
        # info here too: https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/
        # Also in Zisserman's book p. 161-162
        uvPoint = np.ones(3)  # pixel point
        uvPoint[0:2] = pts2D[idx, 0:2]
        ray = Minv.dot(mu * uvPoint - p4)
        rays.append(ray)

    return rays


def shortestDistanceBetweenRays(ray1, ray2):
    # ray1 and ray2 are in the form: [x1, y1, z1, x2, y2, z2]
    # info: https://www.geeksforgeeks.org/shortest-distance-between-two-lines-in-3d-space-class-12-maths/

    ray1 = np.asarray(ray1)
    ray2 = np.asarray(ray2)
    a1 = ray1[0:3]  # position vector of some point on the ray
    a2 = ray2[0:3]   # position vector of some point on the ray
    b1 = ray1[3:] - ray1[0:3]  # vector displacement
    b2 = ray2[3:] - ray2[0:3]  # vector displacement
    #shortest_distance = np.linalg.norm(np.cross(b1, b2) * (a2 - a1)) / np.linalg.norm(np.cross(b1, b2))
    shortest_distance = np.linalg.norm(np.dot(np.cross(b1, b2), (a2 - a1))) / np.linalg.norm(np.cross(b1, b2))
    return shortest_distance


def bubbleMetrics(bbox1, proj_matrix1, intrinsic1, dist_coeffs1, bbox2, proj_matrix2, intrinsic2, dist_coeffs2, vsize_cam='both'):
    # Bubble Centre
    pt1 = (bbox1[0:2] + bbox1[2:4]) / 2  # bubble centre point
    pt2 = (bbox2[0:2] + bbox2[2:4]) / 2  # bubble centre point
    undist_pt1 = cv2.undistortPoints(np.expand_dims(pt1, axis=1), intrinsic1, dist_coeffs1, None, intrinsic1)  # undistort bubble centre point
    undist_pt2 = cv2.undistortPoints(np.expand_dims(pt2, axis=1), intrinsic2, dist_coeffs2, None, intrinsic2)  # undistort bubble centre point
    triangulation_coords4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, undist_pt1, undist_pt2)
    triangulation_coords4d /= triangulation_coords4d[3, :]  # 3D positions in mm. Each column is a point (not rows!)
    xc, yc, zc = triangulation_coords4d[0:3, 0]  # x, y, z coordinates of bubble center

    # Bubble Size and Volume
    bbox1_midpts = [(bbox1[0], pt1[1]), (pt1[0], bbox1[1]), (bbox1[2], pt1[1]), (pt1[0], bbox1[3])]  # middle points of each side of the bbox [left, top, right, bottom]
    bbox2_midpts = [(bbox2[0], pt2[1]), (pt2[0], bbox2[1]), (bbox2[2], pt2[1]), (pt2[0], bbox2[3])]  # middle points of each side of the bbox [left, top, right, bottom]
    undist_bbox1 = cv2.undistortPoints(np.asarray(bbox1_midpts).T, intrinsic1, dist_coeffs1, None, intrinsic1)  # undistort middle points of bbox's sides
    undist_bbox2 = cv2.undistortPoints(np.asarray(bbox2_midpts).T, intrinsic2, dist_coeffs2, None, intrinsic2)  # undistort middle points of bbox's sides
    bbox1_xyz = cv2.triangulatePoints(proj_matrix1, proj_matrix2, np.squeeze(undist_bbox1).T, np.tile(np.squeeze(undist_pt2), (4, 1)).T)
    bbox1_xyz /= bbox1_xyz[3, :]  # 3D positions in mm. Each column is a point (not rows!)
    bbox2_xyz = cv2.triangulatePoints(proj_matrix1, proj_matrix2, np.tile(np.squeeze(undist_pt1), (4, 1)).T, np.squeeze(undist_bbox2).T)
    bbox2_xyz /= bbox2_xyz[3, :]  # 3D positions in mm. Each column is a point (not rows!)

    hsize1 = np.linalg.norm(bbox1_xyz[:, 2] - bbox1_xyz[:, 0])  # horizontal size on LEFT camera [mm]
    hsize2 = np.linalg.norm(bbox2_xyz[:, 2] - bbox2_xyz[:, 0])  # horizontal size on RIGHT camera [mm]
    vsize1 = np.linalg.norm(bbox1_xyz[:, 3] - bbox1_xyz[:, 1])  # vertical size on LEFT camera [mm]
    vsize2 = np.linalg.norm(bbox2_xyz[:, 3] - bbox2_xyz[:, 1])  # vertical size on RIGHT camera [mm]
    if vsize_cam.lower() == 'left':
        vsize = vsize1.copy()  # bubble vertical diameter [mm]
    elif vsize_cam.lower() == 'right':
        vsize = vsize2.copy()  # bubble vertical diameter [mm]
    else:
        vsize = (vsize1 + vsize2) / 2  # bubble vertical diameter [mm]
    bubbleVol_mL = (4 / 3) * np.pi * (hsize1 / 2) * (hsize2 / 2) * (vsize / 2) / 1000  # bubble volume [mL]

    return bubbleVol_mL, (hsize1, hsize2, vsize), (xc, yc, zc), triangulation_coords4d


def populateBubbleArrays(k, bbvol_mL, bbsize, bbcentre, bubbles4D, bubbleData, bubbleDataTemplate, pair, frameNr, fps):
    hsizeL, hsizeR, vsize = bbsize  # bubble diameters [mm]
    eqr = ((hsizeL/2) * (hsizeR/2) * (vsize/2)) ** (1 / 3)  # equivalent sphere radius [mm]
    xc, yc, zc = bbcentre

    ###########################################################################################
    # populate bubbles4D list (see variable initialization for description)
    while k >= bubbles4D.__len__():
        bubbles4D.append(bubbleDataTemplate)

    hsizeL_median = np.nan
    hsizeR_median = np.nan
    vsize_median = np.nan
    eqr_median = np.nan  # sphere equivalent radius
    bbvol_mL_median = np.nan
    rise_velocity_cms = np.nan
    travelled_distance_cm = np.nan
    frame0 = frameNr
    frameCnt = 1  # number of frames in which this bidL-bidR match was detected
    bubbles4D_sel = bubbles4D[k]
    if bubbles4D_sel.__len__() == 0:  # new bubble
        hsizeL_median = hsizeL
        hsizeR_median = hsizeR
        vsize_median = vsize
        eqr_median = eqr
        bbvol_mL_median = bbvol_mL
    elif pair[1] not in bubbles4D_sel[:, 4].astype(np.uint32):  # existing bubble but new matching
        hsizeL_median = hsizeL
        hsizeR_median = hsizeR
        vsize_median = vsize
        eqr_median = eqr
        bbvol_mL_median = bbvol_mL
    else:  # existing bubble and existing pair
        # IMPORTANT: bidL and bidR are different from kL and kR!
        # - bidL, bidR are the bubble IDs given by the two trackers.
        # - kL and kR are the indices of the bidL and bidR bubbles in the bubblesL and bubblesR class objects.
        bidRsel = bubbles4D_sel[:, 4].astype(np.uint32) == pair[1]
        frame0 = bubbles4D_sel[bidRsel, 0][0].astype(np.uint32)  # first frame where bubble match was detected
        frameCnt = np.sum(bidRsel) + 1
        hsizeL_median = np.nanmedian(np.append(bubbles4D_sel[bidRsel, 8], hsizeL))
        hsizeR_median = np.nanmedian(np.append(bubbles4D_sel[bidRsel, 9], hsizeR))
        vsize_median = np.nanmedian(np.append(bubbles4D_sel[bidRsel, 10], vsize))
        eqr_median = np.nanmedian(np.append(bubbles4D_sel[bidRsel, 11], eqr))
        bbvol_mL_median = np.nanmedian(np.append(bubbles4D_sel[bidRsel, 12], bbvol_mL))
        # travelled_distance_cm = (zc - bubbles4D_sel[bidRsel, 7][0]) / 10  # bubble travelled distance [cm]
        travelled_distance_cm = (((xc - bubbles4D_sel[bidRsel, 5][0]) ** 2 +
                                  (yc - bubbles4D_sel[bidRsel, 6][0]) ** 2 +
                                  (zc - bubbles4D_sel[bidRsel, 7][
                                      0]) ** 2) ** .5) / 10  # bubble travelled distance [cm]
        rise_velocity_cms = travelled_distance_cm / ((frameNr - frame0) / fps)  # bubble rise velocity [cm/s]

    cbubble = np.array(
        [frame0, frameNr, frameCnt] +
        list(pair) +
        [xc, yc, zc, hsizeL, hsizeR, vsize, eqr, bbvol_mL] +
        [hsizeL_median, hsizeR_median, vsize_median, eqr_median, bbvol_mL_median] +
        [rise_velocity_cms, travelled_distance_cm],
    )  # data about current bubble

    bubbles4D[k] = np.vstack((bubbles4D[k], cbubble[None, :]))  # list of lists for each bubble [bubbleID_left, bubbleID_right, x, y, z, hsizeL, hsizeR, vsize, frameNr]

    ###########################################################################################
    # populate bubbleData array (see variable initialization for description)
    most_common_bidR = np.bincount(bubbles4D[k][:, 4].astype(
        np.uint32)).argmax()  # find which bidR bubble, the current bidL bubble is most often associated with
    if pair[1] == most_common_bidR:
        bidRsel = bubbles4D[k][:, 4] == most_common_bidR
        bubble_row = bubbles4D[k][bidRsel, :][-1, :]  # bubble_row is not necessarily equal to cbubble
        idx, = np.where(bubbleData[:, 3] == pair[0])
        if idx.__len__() == 0:
            bubbleData = np.vstack((bubbleData, bubble_row))  # append new bubble
        else:
            # WARNING: the number of rows in bubbleData and in bubbles4D are different!!
            # bubbles4D can include empty lists, whereas bubbleData is a numpy array and it cannot have
            # empty rows
            bubbleData[idx, :] = bubble_row  # copy most updated bubble version

    return bubbles4D, bubbleData


def shutterspeedZcorrection(bubbleData, shutterSpeed):
    # Z-correction based on shutterspeed
    rv = bubbleData[:, 18].copy()  # rise velocities
    rv[np.argwhere(np.isnan(rv))] = np.nanmedian(rv)  # replace nan values with median velocity

    bubbleData[:, 15] -= (rv * shutterSpeed) # correct bubble vsize
    bubbleData[:, 16] = ((bubbleData[:, 13]/2) * (bubbleData[:, 14]/2) * (bubbleData[:, 15]/2))**(1/3)  # correct bubble sphere-equivalent radius [mm]
    bubbleData[:, 17] = (4 / 3) * np.pi * (bubbleData[:, 16]**3) / 1000  # correct bubble volume [mL]

    return bubbleData


def flowRate(bubbleData, fps):

    total_bubble_vol = np.nansum(bubbleData[:, 17])  # add bubble volumes
    elapsedtime_sec = (bubbleData[:, 1].max() - bubbleData[:, 0].min()) / fps  # elapsed time between first and last bubbles
    # elapsedtime_sec is not based on the total number of frames to prevent erroneous results if video not cropped
    flowrate = total_bubble_vol / (elapsedtime_sec / 60)  # Flow rate (mL/min)

    return flowrate


def main():
    print("\nBURST: BUbble Rise and Size Tracking\n")

    # Read the path of the script file
    spath = os.path.dirname(sys.argv[0])
    apath = os.path.abspath(spath)

    # Read configuration file if it exists
    configFile = os.path.join(apath, "BURST_config.txt")
    try:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(configFile)
    except IOError:
        print(f"Configuration file missing: {configFile}.")
        sys.exit(1)
    else:
        vpath = config.get('Parameters', 'VideoPath')
        vLfile = config.get('Parameters', 'VideoFileLEFT')
        vRfile = config.get('Parameters', 'VideoFileRIGHT')
        pLfile = config.get('Parameters', 'CameraParametersFileLEFT')
        pRfile = config.get('Parameters', 'CameraParametersFileRIGHT')
        chesspathL = config.get('Parameters', 'ChessboardImagesPathLEFT')
        chesspathR = config.get('Parameters', 'ChessboardImagesPathRIGHT')
        shutterSpeed = config.get('Parameters', 'ManualShutterSpeed_sec')
        if shutterSpeed is not None:
            shutterSpeed = str_to_float(shutterSpeed)
        vsizeCam = config.get('Parameters', 'MainCameraForVerticalBubbleSize').lower()

        # Objects detected in masked areas will be filtered out.
        # Masked areas are given as follow: left, top, right, bottom, left, top, right, bottom, etc...
        # - horizontal pixel coordinates increase towards right
        # - vertical pixel coordinates increase downwards
        maskedAreasL = config.get('Parameters', 'MaskAreasLEFT')
        maskedAreasR = config.get('Parameters', 'MaskAreasRIGHT')
        if maskedAreasL is not None and len(maskedAreasL) > 0:
            maskedAreasL = tuple(map(int, maskedAreasL.split(",")))  # horizontal and vertical coordinates of TL and BR corners of areas to mask
            maskedAreasL = np.reshape(maskedAreasL, (-1, 4))
        if maskedAreasR is not None and len(maskedAreasR) > 0:
            maskedAreasR = tuple(map(int, maskedAreasR.split(",")))  # horizontal and vertical coordinates of TL and BR corners of areas to mask
            maskedAreasR = np.reshape(maskedAreasR, (-1, 4))

        showFrames = config.getboolean('Parameters', 'showFrames')
        exportVideoFromTracking = config.getboolean('Parameters', 'exportVideoFromTracking')

        unsharpMask = config.getboolean('Deblurring', 'UnsharpMask')
        unsharpMask_radius = float(config.get('Deblurring', 'radius'))
        unsharpMask_amount = float(config.get('Deblurring', 'amount'))

        # Numbers of horizontal and vertical checkerboard points (horizontal, vertical)
        # points = (number of squares - 1) because what matters are the corners between 4 squares
        chessboardPoints = tuple(map(int, config.get('Checkerboard', 'checkerboardPoints').split(",")))
        chessboardCellsize_mm = float(config.get('Checkerboard', 'checkerboardCellsize_mm'))

        yolopath = config.get('Detection and Tracking', 'YoloPath')
        yolo_weightsL = os.path.join(yolopath, config.get('Detection and Tracking', 'WeightFileLEFT'))
        yolo_cfgL = os.path.join(yolopath, config.get('Detection and Tracking', 'ConfigurationFileLEFT'))
        yolo_weightsR = os.path.join(yolopath, config.get('Detection and Tracking', 'WeightFileRIGHT'))
        yolo_cfgR = os.path.join(yolopath, config.get('Detection and Tracking', 'ConfigurationFileRIGHT'))
        classes2exclude = config.get('Detection and Tracking', 'YoloClassesToExclude')  # ID number of the object classes that should be filtered out of the analysis (ID number used during YOLO training)
        if classes2exclude is not None and len(classes2exclude) > 0:
            classes2exclude = tuple(map(int, classes2exclude.split(",")))

        sort_max_age = int(config.get('Detection and Tracking', 'SORT_max_age'))  # SORT parameter: Maximum number of frames to keep alive a track without associated detections (SORT default: 1)
        min_hits = int(config.get('Detection and Tracking', 'SORT_min_hits'))  # SORT parameter: Minimum number of associated detections before track is initialised (SORT default: 3)
        iou_threshold = float(config.get('Detection and Tracking', 'SORT_iou_threshold'))  # SORT parameter: Minimum IOU for match (SORT default: 0.3). IOU = intersection-over-union
        rectangle_coefficientL = float(config.get('Detection and Tracking', 'SORT_rectangle_coefficient_LEFT'))  # SORT parameter implemented by YM: coefficient to enlarge the size of the tracker rectangle in order to facilitate tracking when frame rate is too low.
        rectangle_coefficientR = float(config.get('Detection and Tracking', 'SORT_rectangle_coefficient_RIGHT'))  # SORT parameter implemented by YM: coefficient to enlarge the size of the tracker rectangle in order to facilitate tracking when frame rate is too low.

        confidence_threshold = float(config.get('Detection and Tracking', 'ConfidenceThreshold'))  # confidence threshold
        nms_threshold = float(config.get('Detection and Tracking', 'NMS_Threshold'))  # non-maximum suppression threshold

        history_duration_sec = float(config.get('Detection and Tracking', 'HistoryDuration_sec'))  # any bubble last seen more than this number of seconds will be deleted from the bubbles4D variable to save memory and increase speed

        distthreshold = tuple(map(float, config.get('Matching', 'DistanceThresholds').split(",")))  # [min, max] threshold on distance between camera rays for bubble matching [mm]
        bubbleThreshold = int(config.get('Matching', 'BubbleThresholdForFastMatching'))

        DisplayPixelWidth = int(config.get('Display and Export', 'DisplayPixelWidth'))
        DisplayPixelHeight = int(config.get('Display and Export', 'DisplayPixelHeight'))
        ExportPixelWidth = int(config.get('Display and Export', 'ExportPixelWidth'))
        ScreenDPI = int(config.get('Display and Export', 'ScreenDPI'))
        bidLsize = float(config.get('Display and Export', 'BubbleNumberSizeLEFT'))
        bidRsize = float(config.get('Display and Export', 'BubbleNumberSizeRIGHT'))
        Verbose = config.getboolean('Display and Export', 'VerboseMode')

    export_dpi = ScreenDPI * ExportPixelWidth / DisplayPixelWidth
    imext = '.jpg'

    # Creates output directories
    outpath = os.path.join(vpath, "output")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Copy the configuration file to the output directory
    copyfile(configFile, os.path.join(outpath, "_BURST_settings.txt"))

    bubbles4D = []  # list of numpy arrays (one list element per bubble, one numpy array row per frame with the bubble)
    bubbleDataTemplate = np.array([], dtype=np.uint32).reshape(0, 20)
    bubbleData = bubbleDataTemplate.copy()  # numpy array (containing always the most up-to-date bubble stats (one row per bubble)

    # 'Small' and 'Large' variables are used to estimate the error caused by the detection rectangles (Yolo rectangles
    # tend to be too large when bubble edges are fuzzy). The variables have the same structure as the variables defined
    # above.
    bubbles4D_Small = []  # list of numpy arrays (one list element per bubble, one numpy array row per frame with the bubble)
    bubbleData_Small = bubbleDataTemplate.copy()  # numpy array (containing always the most up-to-date bubble stats (one row per bubble)
    bubbles4D_Large = []  # list of numpy arrays (one list element per bubble, one numpy array row per frame with the bubble)
    bubbleData_Large = bubbleDataTemplate.copy()  # numpy array (containing always the most up-to-date bubble stats (one row per bubble)

    # Determine the backend for plotting figures
    matplotlib.use('Agg')  # use a non-interactive backend (Agg for PNGs)
    import matplotlib.pyplot as plt  # the backend needs to be defined before importing pyplot


    ##################################################################################################################
    # GET INTRINSICS AND EXTRINSICS OF BOTH CAMERAS
    ##################################################################################################################
    # Read camera parameters file if it exists
    try:
        calibL = np.loadtxt(os.path.join(apath, 'camera_parameters', pLfile))
    except IOError:
        print(f"LEFT Camera calibration file missing {pLfile}.")
        sys.exit(1)
    else:
        intrinsicL = calibL[0:9].reshape((3, 3))
        distCoeffsL = calibL[9:14]

    try:
        calibR = np.loadtxt(os.path.join(apath, 'camera_parameters', pRfile))
    except IOError:
        print(f"RIGHT Camera calibration file missing {pRfile}.")
        sys.exit(1)
    else:
        intrinsicR = calibR[0:9].reshape((3, 3))
        distCoeffsR = calibR[9:14]

    # Get chessboard coordinates automatically
    print("Estimation of relative camera positions.")
    objPoints, imgPointsL, imgPointsR = get_chessboard_objpoints_and_imgpoints(chesspathL, chesspathR, chessboardPoints, chessboardCellsize_mm, pattern='*.jpg')

    if objPoints.__len__()==0 or imgPointsL.__len__()==0 or imgPointsR.__len__()==0:
        print("\nAutomatic detection of checkerboard corners failed. Manual corner selection required.")
        print("\nSelect all checkerboard corners starting from left-to-right and from top-to-bottom of the checkerboard (not of the image!)\n")
        # Get chessboard coordinates manually
        objPoints, imgPointsL, imgPointsR = select_chessboard_objpoints_and_imgpoints(chesspathL, chesspathR, chessboardPoints, chessboardCellsize_mm, pattern='*.jpg')


    # retval, _, _, _, _, R, T, E, F, perViewErrors = cv2.stereoCalibrateExtended(objPoints, imgPointsL,
    #                                                                                             imgPointsR, intrinsicL,
    #                                                                                             distCoeffsL, intrinsicR,
    #                                                                                             distCoeffsR, None, None,
    #                                                                                             None,
    #                                                                                             flags=cv2.CALIB_FIX_INTRINSIC)
    retval, _, _, _, _, R, T, E, F, rvecsL, tvecsL, perViewErrors = cv2.stereoCalibrateExtended(objPoints, imgPointsL, imgPointsR, intrinsicL, distCoeffsL, intrinsicR, distCoeffsR, None, None, None, flags=cv2.CALIB_FIX_INTRINSIC)
    #
    # retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F, rvecsL, tvecsL, perViewErrors = cv2.stereoCalibrateExtended(objPoints, imgPointsL, imgPointsR, None, None, None, None, frameSize, None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_S1_S2_S3_S4 + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST)
    # retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F, rvecs, tvecs, _ = cv2.stereoCalibrateExtended(objPoints, imgPointsL, imgPointsR, None, None, None, None, frameSize, None, None, flags=cv2.CALIB_FIX_INTRINSIC)

    # Find view that has lowest combined error:
    sumIDs = np.nonzero(np.sum(perViewErrors, axis=1) <= np.percentile(np.sum(perViewErrors, axis=1), 25))  # 25% views with smallest sum
    bestView = sumIDs[0][np.argmin(np.std(perViewErrors[sumIDs], axis=1))]
    # rVecL = rvecsL[bestView]
    # tVecL = tvecsL[bestView]
    _, rVecL, tVecL = cv2.solvePnP(objPoints[bestView], imgPointsL[bestView], intrinsicL, distCoeffsL)
    _, rVecR, tVecR = cv2.solvePnP(objPoints[bestView], imgPointsR[bestView], intrinsicR, distCoeffsR)


    # TRANSFORMATION FROM WORLD COORDINATES TO IMAGE COORDINATES
    # ----------------------------------------------------------
    intrinsicL = np.asarray(intrinsicL)  # intrinsic matrix
    distCoeffsL = np.asarray(distCoeffsL)
    RL, JacL = cv2.Rodrigues(rVecL)  # compute rotation matrix from rotation vectors
    extrinsicL = np.hstack((RL, tVecL))  # 3 x 4 rotation and translation matrix (i.e. pose matrix) to transform from object coordinate system to camera coordinate system
    projMatL = intrinsicL.dot(extrinsicL)  # projection matrix of camera = intrinsic matrix * extrinsic (pose) matrix. Project 3D points to image pixel coordinates.

    intrinsicR = np.asarray(intrinsicR)  # intrinsic matrix
    distCoeffsR = np.asarray(distCoeffsR)
    RR, JacR = cv2.Rodrigues(rVecR)  # compute rotation matrix from rotation vectors
    extrinsicR = np.hstack((RR, tVecR))  # 3 x 4 rotation and translation matrix (i.e. pose matrix) to transform from object coordinate system to camera coordinate system
    projMatR = intrinsicR.dot(extrinsicR)  # projection matrix of camera = intrinsic matrix * extrinsic (pose) matrix. Project 3D points to image pixel coordinates.


    # REVERSE TRANSFORMATION FROM WORLD COORDINATES TO IMAGE COORDINATES
    # ------------------------------------------------------------------
    # camera_position_in_world_coordinates = -(inv(R))*T
    RinvL = RL.transpose()  # inverse of rotation matrix is its transpose
    camL_3D_position = (-RinvL).dot(tVecL)  # '.dot for matrix multiplication

    RinvR = RR.transpose()  # inverse of rotation matrix is its transpose
    camR_3D_position = (-RinvR).dot(tVecR)  # '.dot for matrix multiplication


    # https://stackoverflow.com/questions/16265714/camera-pose-estimation-opencv-pnp
    # Now pos is the position of the camera expressed in the global frame (the same frame the objectPoints are
    # expressed in). R is an attitude matrix DCM which is a good form to store the attitude in.
    # If you require Euler angles then you can convert the DCM to Euler angles given an XYZ rotation sequence using:
    #
    # roll = atan2(-R[2][1], R[2][2])
    # pitch = asin(R[2][0])
    # yaw = atan2(-R[1][0], R[0][0])

    camL_rpy = np.asarray(( np.arctan2(-RL[2, 1], RL[2, 2]),  # roll
                 np.arcsin(RL[2, 0]),  # pitch
                 np.arctan2(-RL[1, 0], RL[0, 0]))) * 180 / np.pi  # yaw
    camR_rpy = np.asarray(( np.arctan2(-RR[2, 1], RR[2, 2]),  # roll
                 np.arcsin(RR[2, 0]),  # pitch
                 np.arctan2(-RR[1, 0], RR[0, 0]))) * 180 / np.pi  # yaw

    ##################################################################################################################
    # PLOT 3D CAMERA POSITIONS
    ##################################################################################################################
    plt.ion()
    width_in_inches = DisplayPixelWidth / ScreenDPI
    height_in_inches = DisplayPixelHeight / ScreenDPI
    fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=ScreenDPI)  # figure size is in inches
    ax = fig.add_subplot(projection='3d')
    #plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)  # this line is needed to make the axes fill the figure
    #set_axes_equal(ax)

    meshX, meshY = np.meshgrid(objPoints[bestView][:, 0], objPoints[bestView][:, 1])
    meshZ = np.zeros(meshX.shape)
    ax.plot_wireframe(meshX, meshY, meshZ, rstride=1, cstride=1, color='0.25', linewidth=1)

    # Plot "world" frame axes
    plt.plot([0, chessboardCellsize_mm * 2], [0, 0], zs=[0, 0], linestyle='solid', color='r', linewidth=3)
    plt.plot([0, 0], [0, chessboardCellsize_mm * 2], zs=[0, 0], linestyle='solid', color='g', linewidth=3)
    plt.plot([0, 0], [0, 0], zs=[0, chessboardCellsize_mm * 2], linestyle='solid', color='b', linewidth=3)

    sc = 20  # scaling factor for camera axes
    # plot camera L axes
    plt.plot([camL_3D_position[0][0], camL_3D_position[0][0] + RL[0, 0]*sc],
             [camL_3D_position[1][0], camL_3D_position[1][0] + RL[0, 1]*sc],
             zs=[camL_3D_position[2][0], camL_3D_position[2][0] + RL[0, 2]*sc], linestyle='solid', color='r', linewidth=1)
    plt.plot([camL_3D_position[0][0], camL_3D_position[0][0] + RL[1, 0]*sc],
             [camL_3D_position[1][0], camL_3D_position[1][0] + RL[1, 1]*sc],
             zs=[camL_3D_position[2][0], camL_3D_position[2][0] + RL[1, 2]*sc], linestyle='solid', color='g', linewidth=1)
    plt.plot([camL_3D_position[0][0], camL_3D_position[0][0] + RL[2, 0]*sc],
             [camL_3D_position[1][0], camL_3D_position[1][0] + RL[2, 1]*sc],
             zs=[camL_3D_position[2][0], camL_3D_position[2][0] + RL[2, 2]*sc], linestyle='solid', color='b', linewidth=1)

    # plot camera R axes
    plt.plot([camR_3D_position[0][0], camR_3D_position[0][0] + RR[0, 0]*sc],
             [camR_3D_position[1][0], camR_3D_position[1][0] + RR[0, 1]*sc],
             zs=[camR_3D_position[2][0], camR_3D_position[2][0] + RR[0, 2]*sc], linestyle='solid', color='r', linewidth=1)
    plt.plot([camR_3D_position[0][0], camR_3D_position[0][0] + RR[1, 0]*sc],
             [camR_3D_position[1][0], camR_3D_position[1][0] + RR[1, 1]*sc],
             zs=[camR_3D_position[2][0], camR_3D_position[2][0] + RR[1, 2]*sc], linestyle='solid', color='g', linewidth=1)
    plt.plot([camR_3D_position[0][0], camR_3D_position[0][0] + RR[2, 0]*sc],
             [camR_3D_position[1][0], camR_3D_position[1][0] + RR[2, 1]*sc],
             zs=[camR_3D_position[2][0], camR_3D_position[2][0] + RR[2, 2]*sc], linestyle='solid', color='b', linewidth=1)

    camCorners = [[-16, -16, 16, 16], [-9, 9, 9, -9], [40, 40, 40, 40]]
    camLcorners = (RinvL).dot(np.asarray(camCorners) - tVecL)
    pyramidX = list(camLcorners[0, :]), \
               list(camL_3D_position[0]) + list(camLcorners[0, [0, 1]]), \
               list(camL_3D_position[0]) + list(camLcorners[0, [1, 2]]), \
               list(camL_3D_position[0]) + list(camLcorners[0, [2, 3]]), \
               list(camL_3D_position[0]) + list(camLcorners[0, [3, 0]])
    pyramidY = list(camLcorners[1, :]), \
               list(camL_3D_position[1]) + list(camLcorners[1, [0, 1]]), \
               list(camL_3D_position[1]) + list(camLcorners[1, [1, 2]]), \
               list(camL_3D_position[1]) + list(camLcorners[1, [2, 3]]), \
               list(camL_3D_position[1]) + list(camLcorners[1, [3, 0]])
    pyramidZ = list(camLcorners[2, :]), \
               list(camL_3D_position[2]) + list(camLcorners[2, [0, 1]]), \
               list(camL_3D_position[2]) + list(camLcorners[2, [1, 2]]), \
               list(camL_3D_position[2]) + list(camLcorners[2, [2, 3]]), \
               list(camL_3D_position[2]) + list(camLcorners[2, [3, 0]])
    surfaces = []
    for i in range(len(pyramidX)):
        surfaces.append([list(zip(pyramidX[i], pyramidY[i], pyramidZ[i]))])
    for i, surface in enumerate(surfaces):
        if i == 0:
            fc = 'yellow'
            ec = None
            alpha = None
        else:
            fc = 'green'
            ec = '0.8'
            alpha = 0.5

        ax.add_collection3d(Poly3DCollection(surface, facecolors=fc, linewidths=1, edgecolors=ec, alpha=alpha))
        #ax.add_collection3d(Poly3DCollection(surface, facecolors=fc, linewidths=1, edgecolors='r', alpha=.20))

    camRcorners = (RinvR).dot(np.asarray(camCorners) - tVecR)
    pyramidX = list(camRcorners[0, :]), \
               list(camR_3D_position[0]) + list(camRcorners[0, [0, 1]]), \
               list(camR_3D_position[0]) + list(camRcorners[0, [1, 2]]), \
               list(camR_3D_position[0]) + list(camRcorners[0, [2, 3]]), \
               list(camR_3D_position[0]) + list(camRcorners[0, [3, 0]])
    pyramidY = list(camRcorners[1, :]), \
               list(camR_3D_position[1]) + list(camRcorners[1, [0, 1]]), \
               list(camR_3D_position[1]) + list(camRcorners[1, [1, 2]]), \
               list(camR_3D_position[1]) + list(camRcorners[1, [2, 3]]), \
               list(camR_3D_position[1]) + list(camRcorners[1, [3, 0]])
    pyramidZ = list(camRcorners[2, :]), \
               list(camR_3D_position[2]) + list(camRcorners[2, [0, 1]]), \
               list(camR_3D_position[2]) + list(camRcorners[2, [1, 2]]), \
               list(camR_3D_position[2]) + list(camRcorners[2, [2, 3]]), \
               list(camR_3D_position[2]) + list(camRcorners[2, [3, 0]])
    surfaces = []
    for i in range(len(pyramidX)):
        surfaces.append([list(zip(pyramidX[i], pyramidY[i], pyramidZ[i]))])
    for i, surface in enumerate(surfaces):
        if i == 0:
            fc = 'yellow'
            ec = None
            alpha = None
        else:
            fc = 'cyan'
            ec = '0.8'
            alpha = 0.5

        ax.add_collection3d(Poly3DCollection(surface, facecolors=fc, linewidths=1, edgecolors=ec, alpha=alpha))

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_zlim3d(bottom=0)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)  # this line is needed to make the axes fill the figure
    set_axes_equal(ax)
    fig.savefig(os.path.join(outpath, f'Camera_positions_3D') + imext, dpi=export_dpi)  # exports the figure to a jpg or PNG file

    ax.view_init(elev=0, azim=0, roll=90)
    #fig.canvas.draw()  # convert canvas to image
    fig.savefig(os.path.join(outpath, f'Camera_positions_YZ') + imext, dpi=export_dpi)  # exports the figure to a jpg or PNG file

    ax.view_init(elev=0, azim=-90, roll=0)
    #fig.canvas.draw()  # convert canvas to image
    fig.savefig(os.path.join(outpath, f'Camera_positions_XZ') + imext, dpi=export_dpi)  # exports the figure to a jpg or PNG file

    ax.view_init(elev=90, azim=-90, roll=0)
    # fig.canvas.draw()  # convert canvas to image
    fig.savefig(os.path.join(outpath, f'Camera_positions_XY') + imext, dpi=export_dpi)  # exports the figure to a jpg or PNG file

    ax.view_init()  # reset 3D view
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)  # this line is needed to make the axes fill the figure
    set_axes_equal(ax)

    ##################################################################################################################

    # Initialization
    capL = cv2.VideoCapture(os.path.join(vpath, vLfile))
    capR = cv2.VideoCapture(os.path.join(vpath, vRfile))

    # define video variables
    nbFramesL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    nbFramesR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
    fpsL = capL.get(cv2.CAP_PROP_FPS)
    fpsR = capR.get(cv2.CAP_PROP_FPS)
    fps = min(fpsL, fpsR)  # min frame rate (used for flow rate calculations)

    vduration = min(nbFramesL / fpsL, nbFramesR / fpsR)  # duration of shortest video [sec]
    if fpsL == fpsR:
        frameMax = min(nbFramesL, nbFramesR)
        if vsizeCam is None:
            vsizeCam = 'both'
    elif fpsL < fpsR:
        frameMax = min(nbFramesL, int(vduration * fpsL))
        if vsizeCam is None or vsizeCam == 'both':  # cannot use both cameras for vsize if framerates are different
            vsizeCam = 'right'  # use camera with fastest frame rate to assess vertical bubble dimension
    else:
        frameMax = min(nbFramesR, int(vduration * fpsR))
        if vsizeCam is None or vsizeCam == 'both':  # cannot use both cameras for vsize if framerates are different
            vsizeCam = 'left'  # use camera with fastest frame rate to assess vertical bubble dimension

    videoSize = (
        int(capL.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    if shutterSpeed is None or shutterSpeed <= 0:
        if vsizeCam.lower() == 'left':
            shutterSpeed = 1. / (2 * fpsL)
        elif vsizeCam.lower() == 'right':
            shutterSpeed = 1. / (2 * fpsR)
        else:  # cameras have same framerate
            shutterSpeed = 1. / (2 * fps)  # use shutter speed of camera with highest framerate


    # INITIALIZE YOLO NETWORK
    # -----------------------
    netL = cv2.dnn.readNet(yolo_weightsL, yolo_cfgL)  # Load Yolo
    netR = cv2.dnn.readNet(yolo_weightsR, yolo_cfgR)  # Load Yolo

    # Enable GPU CUDA
    netL.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netL.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    netR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names_L = netL.getLayerNames()
    layer_names_R = netR.getLayerNames()
    # output_layers_L = [layer_names_L[i[0] - 1] for i in netL.getUnconnectedOutLayers()]
    # output_layers_R = [layer_names_R[i[0] - 1] for i in netR.getUnconnectedOutLayers()]
    output_layers_L = [layer_names_L[i - 1] for i in netL.getUnconnectedOutLayers()]
    output_layers_R = [layer_names_R[i - 1] for i in netR.getUnconnectedOutLayers()]

    # SORT TRACKER
    # -----------------------
    from sort_mod import Sort
    trackerL = Sort(max_age=sort_max_age, min_hits=min_hits, iou_threshold=iou_threshold, rectangle_coefficient=rectangle_coefficientL)  # create instance of the SORT tracker
    trackerR = Sort(max_age=sort_max_age, min_hits=min_hits, iou_threshold=iou_threshold, rectangle_coefficient=rectangle_coefficientR)  # create instance of the SORT tracker

    # max_age: Maximum number of frames to keep alive a track without associated detections (SORT default: 1)
    # min_hits: Minimum number of associated detections before track is initialised (SORT default: 3)
    # iou_threshold: Minimum IOU for match (SORT default: 0.3). IOU = intersection-over-union
    # rectangle_coefficient (Parameter implemented by YM): coefficient to enlarge the size of the detected and the
    # estimated rectangles in order to facilitate tracking when frame rate is too low and that detected rectangles
    # do not overlap with rectangles estimated by the tracker.

    # MAIN LOOP
    # ---------
    # Initialization
    kL_offset = 0  # index offset used for bubbles4D variable
    bubblesL = Bubbles()  # Initialize Bubble elements (lists of bubbles) on left-hand camera
    bubblesR = Bubbles()  # Initialize Bubble elements (lists of bubbles) on right-hand camera

    print(f'\n##############################################################')
    print('Processing video frames...')
    frameNr = 0
    frameNr_high_fps = 0
    while frameNr <= frameMax:
        print(f'{frameNr}/{frameMax}')

        if fpsL == fpsR:
            successL, frameL = capL.read()
            successR, frameR = capR.read()
        elif fpsL < fpsR:
            successL, frameL = capL.read()
            while frameNr_high_fps <= round(frameNr * fpsR / fpsL):
                successR, frameR = capR.read()
                frameNr_high_fps += 1
                if successR:  # update tracker for video with higher framerate
                    frame_tmp = frameR.copy()
                    if unsharpMask:
                        frame_tmp = np.uint8(np.round(unsharp_mask(frame_tmp, radius=unsharpMask_radius, amount=unsharpMask_amount)*255, decimals=0))  # Unsharp mask
                    bbsR, classlistR = find_objects(netR, frame_tmp, output_layers_R, confidence_threshold, nms_threshold)
                    bbsR, classlistR = filter_by_class(bbsR, classlistR, classes2exclude)
                    bbsR = filter_by_position(bbsR, maskedAreasR)
                    trackerR, _, _ = update_tracker(trackerR, bbsR, frame_tmp, fontsize=bidRsize)

        else:
            successR, frameR = capR.read()
            while frameNr_high_fps <= round(frameNr * fpsL / fpsR):
                successL, frameL = capL.read()
                frameNr_high_fps += 1
                if successL:  # update tracker for video with higher framerate
                    frame_tmp = frameL.copy()
                    if unsharpMask:
                        frame_tmp = np.uint8(np.round(unsharp_mask(frame_tmp, radius=unsharpMask_radius, amount=unsharpMask_amount)*255, decimals=0))  # Unsharp mask
                    bbsL, classlistL = find_objects(netL, frame_tmp, output_layers_L, confidence_threshold, nms_threshold)
                    bbsL, classlistL = filter_by_class(bbsL, classlistL, classes2exclude)
                    bbsL = filter_by_position(bbsL, maskedAreasL)
                    trackerL, _, _ = update_tracker(trackerL, bbsL, frame_tmp, fontsize=bidLsize)

        if successL and successR:
            # Unsharp mask
            if unsharpMask:
                frameL = np.uint8(np.round(unsharp_mask(frameL, radius=unsharpMask_radius, amount=unsharpMask_amount)*255, decimals=0))  # Unsharp mask
                frameR = np.uint8(np.round(unsharp_mask(frameR, radius=unsharpMask_radius, amount=unsharpMask_amount)*255, decimals=0))  # Unsharp mask

            # Detect bubbles
            bbsL, classlistL = find_objects(netL, frameL, output_layers_L, confidence_threshold, nms_threshold)
            bbsR, classlistR = find_objects(netR, frameR, output_layers_R, confidence_threshold, nms_threshold)
            #print("Debug: Objects detection completed")

            # if frameNr >= 3:
            #     print("pause")

            # filter out detected objects depending on classes
            bbsL, classlistL = filter_by_class(bbsL, classlistL, classes2exclude)
            bbsR, classlistR = filter_by_class(bbsR, classlistR, classes2exclude)

            # filter out detected objects depending on position in frame
            bbsL = filter_by_position(bbsL, maskedAreasL)
            bbsR = filter_by_position(bbsR, maskedAreasR)

            # Draw masked areas on frames
            frameL = mask_frame(frameL, maskedAreasL, alpha=0.5)
            frameR = mask_frame(frameR, maskedAreasR, alpha=0.5)

            # Update tracker and plot rectangles on frame
            trackerL, tracksL, frameL = update_tracker(trackerL, bbsL, frameL, fontsize=bidLsize)
            trackerR, tracksR, frameR = update_tracker(trackerR, bbsR, frameR, fontsize=bidRsize)

            # Filter out tracked bubbles depending on time (before bubble matching!)
            # This step removes bubble ids that should not be detected anymore because they are too old.
            tracksL = filter_by_time(tracksL, bubblesL, frameNr, fps, (history_duration_sec - 0.5))  # '0.5 sec' substracted to prevent rounding issues (see below)
            tracksR = filter_by_time(tracksR, bubblesR, frameNr, fps, (history_duration_sec - 0.5))  # '0.5 sec' substracted to prevent rounding issues (see below)
            #  Rounding issue causes an 'list index out of range' error later in the code. This occurs when a bubble
            #  that was removed from the 'bubbles4D' variable is somehow detected again and fails to be filtered in
            #  this 'filter_by_time' step. To prevent this, the time threshold used here is set to a smaller value than
            #  the one used to delete bubbles from the 'bubbles4D' variable.

            # Populate bubble lists
            bubblesL.add_frame(frameNr, tracksL)
            bubblesR.add_frame(frameNr, tracksR)

            bidsL = [int(i) for i in tracksL[:, -1]]  # make bubble ids integers
            bidsR = [int(i) for i in tracksR[:, -1]]  # make bubble ids integers
            possible_pairs = list(itertools.product(bidsL, bidsR))
            raysL = np.zeros((possible_pairs.__len__(), 6))
            raysR = np.zeros((possible_pairs.__len__(), 6))
            mindists = np.zeros(possible_pairs.__len__())  # shortest distance between rays
            for k, pair in enumerate(possible_pairs):
                kL = bubblesL.bids.index(pair[0])
                kR = bubblesR.bids.index(pair[1])
                bboxL = np.asarray(bubblesL.bboxes[kL])
                bboxR = np.asarray(bubblesR.bboxes[kR])
                ptL = (bboxL[0:2] + bboxL[2:4]) / 2
                ptR = (bboxR[0:2] + bboxR[2:4]) / 2
                undistCoordsL = cv2.undistortPoints(np.expand_dims(ptL, axis=1), intrinsicL, distCoeffsL, None, intrinsicL)  # undistort points
                undistCoordsR = cv2.undistortPoints(np.expand_dims(ptR, axis=1), intrinsicR, distCoeffsR, None, intrinsicR)  # undistort points

                raysL[k, 0:3] = np.squeeze(camL_3D_position)
                raysR[k, 0:3] = np.squeeze(camR_3D_position)
                raysL[k, 3:] = np.asarray(CameraRay(undistCoordsL, intrinsicL, extrinsicL, mu=60))
                raysR[k, 3:] = np.asarray(CameraRay(undistCoordsR, intrinsicR, extrinsicR, mu=60))

                mindists[k] = shortestDistanceBetweenRays(raysL[k], raysR[k])  # shortest distance between both rays

            keepsel, = np.where((mindists >= distthreshold[0]) * (mindists <= distthreshold[1]))
            if keepsel.__len__() > 0:
                # Keep pairs with mindists shorter than distance threshold
                mindists = mindists[keepsel]
                possible_pairs = [possible_pairs[i] for i in keepsel]

                if possible_pairs.__len__() > bubbleThreshold:  # if too many combinations
                    # then use a different matching technique based on the pairs that have a mindists that is closest
                    # to the median mindist
                    sorted_ind = np.argsort(mindists - np.median(mindists))
                    keepsel = []
                    bidsL_tmp = []
                    bidsR_tmp = []
                    for idx in sorted_ind:
                        if (possible_pairs[idx][0] not in bidsL_tmp) and (possible_pairs[idx][1] not in bidsR_tmp):  # if no duplicated image
                            keepsel += [idx]
                            bidsL_tmp += [possible_pairs[idx][0]]
                            bidsR_tmp += [possible_pairs[idx][1]]

                    # Update mindists and possible_pairs
                    mindists = mindists[keepsel]
                    possible_pairs = [possible_pairs[i] for i in keepsel]

                # Try all bubble matching combinations and keep the solution with the least standard deviation of the
                # mindists. This is not necessarily the solution with the lowest average mindist value. The lowest
                # standard deviation is a more reliable indicator because it assumes that the distance error between
                # rays is similar for all bubbles.
                bidsL_unique = set([x[0] for x in possible_pairs])  # 'set() is used to get unique values
                bidsR_unique = set([x[1] for x in possible_pairs])  # 'set() is used to get unique values
                matchmax = min(len(bidsL_unique), len(bidsR_unique))
                bidsLR = []
                bidsLR_mindists = []
                mindists_std = []
                while bidsLR.__len__() == 0:
                    comblist = list(itertools.combinations(possible_pairs, r=matchmax))
                    comb_mindists = list(itertools.combinations(mindists, r=matchmax))
                    for idx, comb in enumerate(comblist):
                        bidsL_tmp = [x[0] for x in comb]
                        bidsR_tmp = [x[1] for x in comb]
                        if len(bidsL_tmp) != len(set(bidsL_tmp)) or len(bidsR_tmp) != len(set(bidsR_tmp)):  # check for duplicates
                            continue
                        else:  # no duplicated bubble
                            bidsLR.append(comb)
                            bidsLR_mindists.append(comb_mindists[idx])
                            mindists_std.append(np.std(np.array(bidsLR_mindists)))

                    if bidsLR.__len__() == 0:
                        matchmax -= 1

                # Best combination = combination with highest number of matches and lowest error standard deviation
                bestk = np.asarray(mindists_std).argmin()
                if Verbose:
                    print(f'Best matches (frame # {frameNr}):')
                    print(f'Error standard deviation: {mindists_std[bestk]:.3f}')
                bidsLR_Best = bidsLR[bestk]
                for idpair, pair in enumerate(bidsLR_Best):
                    #print(f'Matched pair: {pair[0]} - {pair[1]}: {bidsLR_mindists[bestk][idpair]:.3f}')

                    # calculate 3D position and dimensions of matched bubbles
                    kL = bubblesL.bids.index(pair[0])
                    kR = bubblesR.bids.index(pair[1])
                    bboxL = np.asarray(bubblesL.bboxes[kL])
                    bboxR = np.asarray(bubblesR.bboxes[kR])
                    bbvol_mL, bbsize, bbcentre, triangCoords4D = bubbleMetrics(bboxL, projMatL, intrinsicL, distCoeffsL, bboxR, projMatR, intrinsicR, distCoeffsR, vsize_cam=vsizeCam)

                    # estimate error on dimensions of matched bubbles
                    npxl = 2
                    small_bboxL = np.asarray([bboxL[0] + npxl, bboxL[1] + npxl, bboxL[2] - npxl, bboxL[3] - npxl, bboxL[4]])
                    small_bboxR = np.asarray([bboxR[0] + npxl, bboxR[1] + npxl, bboxR[2] - npxl, bboxR[3] - npxl, bboxR[4]])
                    small_bbvol_mL, small_bbsize, _, _ = bubbleMetrics(small_bboxL, projMatL, intrinsicL, distCoeffsL,
                                                                       small_bboxR, projMatR, intrinsicR, distCoeffsR,
                                                                       vsize_cam=vsizeCam)
                    large_bboxL = np.asarray([bboxL[0] - npxl, bboxL[1] - npxl, bboxL[2] + npxl, bboxL[3] + npxl, bboxL[4]])
                    large_bboxR = np.asarray([bboxR[0] - npxl, bboxR[1] - npxl, bboxR[2] + npxl, bboxR[3] + npxl, bboxR[4]])
                    large_bbvol_mL, large_bbsize, _, _ = bubbleMetrics(large_bboxL, projMatL, intrinsicL, distCoeffsL,
                                                                       large_bboxR, projMatR, intrinsicR, distCoeffsR,
                                                                       vsize_cam=vsizeCam)

                    if Verbose:
                        eqr = ((bbsize[0]/2) * (bbsize[1]/2) * (bbsize[2]/2)) ** (1 / 3)  # equivalent sphere radius [mm]
                        print(f'Matched pair: {pair[0]} - {pair[1]} (dist: {bidsLR_mindists[bestk][idpair]:.2f} mm) | '
                              f'Bubble radii: {eqr:.2f} mm | '
                              f'Bubble volume: {bbvol_mL:.3f} mL')

                    bubbles4D, bubbleData = populateBubbleArrays(kL + kL_offset, bbvol_mL, bbsize, bbcentre, bubbles4D, bubbleData, bubbleDataTemplate, pair, frameNr, fps)
                    bubbles4D_Small, bubbleData_Small = populateBubbleArrays(kL + kL_offset, small_bbvol_mL, small_bbsize, bbcentre, bubbles4D_Small, bubbleData_Small, bubbleDataTemplate, pair, frameNr, fps)
                    bubbles4D_Large, bubbleData_Large = populateBubbleArrays(kL + kL_offset, large_bbvol_mL, large_bbsize, bbcentre, bubbles4D_Large, bubbleData_Large, bubbleDataTemplate, pair, frameNr, fps)

            if showFrames:
                twoFrames = stack_frames(frameL, frameR, direction='horizontal')
                twoFrames = cv2.resize(twoFrames, (1920, 1920*twoFrames.shape[0]//twoFrames.shape[1]))
                cv2.imshow("Frame", twoFrames)
                key = cv2.waitKey()
                #key = cv2.waitKey(1)
                if key == ord("q"):
                    break

            if exportVideoFromTracking:
                if frameNr == 0:  # initialize the output video file
                    vidoutSize = [ExportPixelWidth, ExportPixelWidth * videoSize[1] // (videoSize[0] * 2)]
                    outfileLR = os.path.join(outpath, '_LR_video.mp4')
                    vidoutLR = cv2.VideoWriter(outfileLR, cv2.VideoWriter_fourcc(*'mp4v'), fps, vidoutSize)
                frameLR = stack_frames(frameL, frameR, direction='horizontal')
                frameLR = cv2.resize(frameLR, vidoutSize)
                vidoutLR.write(frameLR)  # write frame in output video

        # Display flow rate of the last X seconds during analysis
        window_duration_sec = 60
        window_frames = round_to_int(fps * window_duration_sec, 0)
        window_frames = max(window_frames, 2)  # prevent errors
        if (frameNr % window_frames) == 0 and frameNr > 0:
            bibLsel = (bubbleData[:, 0] >= (frameNr - window_frames)) * (bubbleData[:, 0] < frameNr)
            window_vol = np.nansum(bubbleData[bibLsel, 17])  # add bubble median volume
            flowrate = window_vol / (window_duration_sec / 60)
            print(f'\n##############################################################')
            print(f'Flow rate during last {window_duration_sec} seconds: {flowrate:.2f} mL/min')
            print(f'##############################################################\n')

        # clean up bubble4D variable otherwise it can become extremely large
        idx, = np.where((frameNr - bubbleData[:, 1]) / fps > history_duration_sec)
        if idx.__len__() > 0:
            bidL_max = bubbleData[idx.max(), 3]
            if bubblesL.bids.index(bidL_max) > abs(kL_offset):  # prevent del_offset from being negative
                kL_max = bubblesL.bids.index(bidL_max)
                del_offset = kL_max + kL_offset
                bubbles4D = bubbles4D[del_offset:]
                bubbles4D_Small = bubbles4D_Small[del_offset:]
                bubbles4D_Large = bubbles4D_Large[del_offset:]
                kL_offset -= del_offset

        frameNr += 1

    # Filter out duplicated bidR bubbles
    # Sometimes multiple bidL get associated with a same bidR. This happens mostly when the tracking looses a bubble for
    # a few frames and gives it a new bid number when it detects it again. This 'new' bidL bubble gets matched to the
    # same bidR bubble, causing the real bubble to be counted twice or more. By filtering our duplicated bidR bubbles
    # from the bubbleData variable, we can limit double counting of a same bubble.
    u, c = np.unique(bubbleData[:, 4], return_counts=True)  # get number of bidR duplicates
    dup = u[c > 1]
    rows_to_delete = []
    for bidR in dup:
        bidRsel = np.where(bubbleData[:, 4] == bidR)
        bestbid = bubbleData[bidRsel, 2].argmax()  # note: argmax only returns the first occurrence
        rows_to_delete += np.delete(bidRsel, bestbid).tolist()
    bubbleData = np.delete(bubbleData, rows_to_delete, axis=0)
    bubbleData_Small = np.delete(bubbleData_Small, rows_to_delete, axis=0)
    bubbleData_Large = np.delete(bubbleData_Large, rows_to_delete, axis=0)

    # Z-correction based on shutter speed
    bubbleData = shutterspeedZcorrection(bubbleData, shutterSpeed)
    bubbleData_Small = shutterspeedZcorrection(bubbleData_Small, shutterSpeed)
    bubbleData_Large = shutterspeedZcorrection(bubbleData_Large, shutterSpeed)

    # Export bubble data
    outfile = os.path.join(outpath, '_bubble_data.txt')
    with open(outfile, 'w') as cfile:
        headerline = 'secStart secStop frameFirst frameLast frameCount bubbleID_L bubbleID_R ' \
                     'bubble_radius_mm bubble_volume_mL ' \
                     'rise_velocity_cm_per_sec travelled_distance_cm ' \
                     'min_radius_mm min_volume_mL ' \
                     'max_radius_mm max_volume_mL'
        secStartStop = bubbleData[:, 0:2] / fps
        export_fmt = '%.3f %.3f %d %d %d %d %d %.3f %.4f %.2f %.2f %.3f %.4f %.3f %.4f'
        export_data = np.column_stack((secStartStop, bubbleData[:, 0:5], bubbleData[:, 16:20],
                                       bubbleData_Small[:, 16:18], bubbleData_Large[:, 16:18]))
        np.savetxt(cfile, export_data, header=headerline, fmt=export_fmt, newline='\n', comments='')

    # Number of bubbles detected
    bubbleCnt = bubbleData.shape[0]
    print(f'\n##############################################################')
    print(f'Number of bubbles detected: {bubbleCnt}')

    # Flow rate (mL/min)
    flowrate = flowRate(bubbleData, fps)
    flowrate_Small = flowRate(bubbleData_Small, fps)
    #flowrate_Large = flowRate(bubbleData_Large, fps)  # 'flowrate' tends to always overestimate the actual flow rates
    # because the YOLO detection boxes are generally larger than the bubbles, but they are rarely smaller. For this
    # reason, 'flowrate_Large' may be ignored.
    print(f'Estimated min/max flow rates: [{flowrate_Small:.2f} mL/min, {flowrate:.2f} mL/min]')

    # Where to find output files
    print(f'\nOUTPUT FILES ARE SAVED IN: {outpath}')
    print(f'##############################################################\n')

    # Export global flow data
    outfile = os.path.join(outpath, '_flow_data.txt')
    with open(outfile, 'w') as cfile:
        cfile.write(f'Number of bubbles detected: {bubbleCnt}\n')
        cfile.write(f'Estimated min/max flow rates: [{flowrate_Small:.2f} mL/min, {flowrate:.2f} mL/min]\n')

    capL.release()
    capR.release()
    if exportVideoFromTracking:
        vidoutLR.release()
    cv2.destroyAllWindows()

    print("\nProcess finished.\n")


if __name__ == '__main__':
    main()
