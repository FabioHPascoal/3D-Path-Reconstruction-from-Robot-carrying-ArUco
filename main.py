import numpy as np
import matplotlib.pyplot as plt
import math
import types
from types import NoneType
import json
import cv2 as cv
from cv2 import aruco

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

# Function to find the center of the aruco by a single camera in each frame
def aruco_center_positions(file_name):

    aruco_dict = aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    vid = cv.VideoCapture(file_name)

    aruco_center_pos = []

    while True:
       
        ret, img = vid.read()
        
        if img is None:
            print("Empty Frame or End of Video")
            break

        corners, ids, rejectedImgPoints = detector.detectMarkers(img)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv.imshow('output', frame_markers)

        if cv.waitKey(1) == ord('q'):
            break

        if (len(corners) > 0) and (0 in ids):

            id0_idx = 0
            if len(ids) > 1:
                id0_idx = np.where(ids == 0)[0][0]

            x_sum = corners[id0_idx][0][0][0]+ corners[id0_idx][0][1][0]+ corners[id0_idx][0][2][0]+ corners[id0_idx][0][3][0]
            y_sum = corners[id0_idx][0][0][1]+ corners[id0_idx][0][1][1]+ corners[id0_idx][0][2][1]+ corners[id0_idx][0][3][1]
                
            x_centerPixel = x_sum*.25
            y_centerPixel = y_sum*.25
            
            aruco_center_pos.append(np.array([[x_centerPixel, y_centerPixel, 1]]))
        
        else:
            aruco_center_pos.append(None)

    return aruco_center_pos

def build_P_matrix(K, R, T):
    projection_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    extrinsic_matrix = np.append(np.append(R, T, axis = 1), np.array([[0, 0, 0, 1]]), axis = 0)

    P = K @ projection_matrix @ np.linalg.inv(extrinsic_matrix)

    return P

if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Load cameras parameters
    K0, R0, T0, res0, dis0 = camera_parameters('0.json')
    K1, R1, T1, res1, dis1 = camera_parameters('1.json')
    K2, R2, T2, res2, dis2 = camera_parameters('2.json')
    K3, R3, T3, res3, dis3 = camera_parameters('3.json')

    P_matrices = []
    P_matrices.append(build_P_matrix(K0, R0, T0))
    P_matrices.append(build_P_matrix(K1, R1, T1))
    P_matrices.append(build_P_matrix(K2, R2, T2))
    P_matrices.append(build_P_matrix(K3, R3, T3))

    aruco_positions = []
    aruco_positions.append(aruco_center_positions("camera-00.mp4"))
    aruco_positions.append(aruco_center_positions("camera-01.mp4"))
    aruco_positions.append(aruco_center_positions("camera-02.mp4"))
    aruco_positions.append(aruco_center_positions("camera-03.mp4"))

    frame_count = len(aruco_positions[0])

    x3d = []
    y3d = []
    z3d = []
    
    for frame in range(frame_count):
        added_points = 0
        for cam in range(4):
            if not isinstance((aruco_positions[cam][frame]), NoneType):
                if added_points == 0:
                    B_matrix = np.append(P_matrices[cam], -1 * aruco_positions[cam][frame].T, axis = 1)

                else:
                    B_matrix = np.append(B_matrix, np.zeros((3 * added_points, 1)), axis = 1)
                    new_lines = np.concatenate((P_matrices[cam], np.zeros((3, added_points)), -1 * aruco_positions[cam][frame].T), axis = 1)
                    B_matrix = np.append(B_matrix, new_lines, axis = 0)

                added_points = added_points + 1

        # Perform SVD(A) = U.S.Vt
        U, S, Vt = np.linalg.svd(B_matrix)

        # Reshape last column of V as the 3 dimension point
        V_last_column = Vt[len(Vt)-1]
        point_3D = V_last_column[:4]
        point_3D = point_3D/point_3D[3]

        x3d.append(point_3D[0])
        y3d.append(point_3D[1])
        z3d.append(point_3D[2])
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3d, y3d, z3d)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()