"""
3D pose estimation with tf open pose and pyopenGL
"""

import os
current_file_path = __file__
current_file_dir = os.path.dirname(__file__)

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import ast

import argparse
import logging

import cv2
import time
import os

from lifting import config

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
import common

logger = logging.getLogger('3D-TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Terrain')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gy.rotate(90, 1, 0, 0)
        gx.translate(-10, 0, 0)
        gy.translate(0, -10, 0)
        gz.translate(0, 0, -10)
        self.window.addItem(gx)
        self.window.addItem(gy)
        self.window.addItem(gz)

        parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        parser.add_argument('--camera', type=int, default=0)
        parser.add_argument('--zoom', type=float, default=1.0)
        parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
        parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
        parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
        args = parser.parse_args()

        logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

        self.lines = {}
        self.connection = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
            [12, 13], [8, 14], [14, 15], [15, 16]
        ]

        w, h = model_wh(args.resolution)
        self.e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        self.cam = cv2.VideoCapture(args.camera)
        logger.info('3d lifting initialization.')
        self.poseLifting = Prob3dPose(os.path.join(current_file_dir,"lifting/models/prob_model_params.mat")
        
        onImage = False


        while onImage == False:
            try:
                ret_val, image = self.cam.read()
                keypoints = self.mesh(image)
                if keypoints.any():
                    onImage = True
            except AssertionError:
                print('body not in image')
            except Exception as ex:
                print(ex)
        

        self.points = gl.GLScatterPlotItem(
            pos=keypoints,
            color=pg.glColor((0, 255, 0)),
            size=15
        )
        self.window.addItem(self.points)

        for n, pts in enumerate(self.connection):
            self.lines[n] = gl.GLLinePlotItem(
                pos=np.array([keypoints[p] for p in pts]),
                color=pg.glColor((0, 0, 255)),
                width=3,
                antialias=True
            )
            self.window.addItem(self.lines[n])

    def mesh(self, image):
        image_h, image_w = image.shape[:2]
        width = 640
        height = 480
        pose_2d_mpiis = []
        visibilities = []

        humans = self.e.inference(image, scales=[None])
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


        cv2.putText(image,
                    "test",
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            return

        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(
        pose_2d_mpiis, visibilities)

        try:
            pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        except Exception as ex:
            raise Exception('Not enough 2D joints identified to generate 3D pose')


        keypoints = pose_3d[0].transpose()

        return keypoints / 80

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            keypoints = self.mesh(image)
        except AssertionError:
            print('body not in image')
        except Exception as ex:
            print(ex)
            return
        else:
            self.points.setData(pos=keypoints)

            for n, pts in enumerate(self.connection):
                self.lines[n].setData(
                    pos=np.array([keypoints[p] for p in pts])
                )

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self, frametime=10):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()


if __name__ == '__main__':
    os.chdir('..')
    t = Terrain()
    t.animation()
