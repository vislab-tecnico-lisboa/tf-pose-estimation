import argparse
import logging
import time

import cv2
import numpy as np
import sys
sys.path.append("/home/avelino/3d_pose/tfopenpose/tf-pose-estimation/src/baseline3d")
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import tensorflow as tf
from baseline3d.predict_3dpose import create_model
import data_utils
import viz

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

map_openpose_baseline = {}

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def converd_skeleton_human(xy,image_w,image_h):
    joints_array = np.zeros((1, 36))
    joints_array[0] = [0 for i in range(36)]
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]
    for o in range(18):
        #feed array with xy array
        try:
            joints_array[0][2*o] = xy[o][0] * image_w
            joints_array[0][2*o + 1] = xy[o][1] * image_h
        except Exception as e:
            joints_array[0][2*o] = 0
            joints_array[0][2*o + 1] = 0
            print('lol')


    _data = joints_array[0]
    # mapping all body parts or 3d-pose-baseline format
    for i in range(len(order)):
        for j in range(2):
            # create encoder input
            enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]

    for j in range(2):
        # Hip
        enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
        # Neck/Nose
        enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
        # Thorax
        enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

    # set spine
    spine_x = enc_in[0][24]
    spine_y = enc_in[0][25]

    enc_in = enc_in[:, dim_to_use_2d]
    mu = data_mean_2d[dim_to_use_2d]
    stddev = data_std_2d[dim_to_use_2d]
    enc_in = np.divide((enc_in - mu), stddev)

    dp = 1.0
    dec_out = np.zeros((1, 48))
    dec_out[0] = [0 for i in range(48)]
    return (enc_in,dec_out,dp)
    """
    enc_in = enc_in[:, dim_to_use_2d]
    mu = data_mean_2d[dim_to_use_2d]
    stddev = data_std_2d[dim_to_use_2d]
    enc_in = np.divide((enc_in - mu), stddev)

    dp = 1.0
    dec_out = np.zeros((1, 48))
    dec_out[0] = [0 for i in range(48)]
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    session3d = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1},
           allow_soft_placement=True))
    batch_size = 128


    actions = data_utils.define_actions( FLAGS.action )
    op3dgraph = create_model(session3d, batch_size)

    ###FOR PLOTING
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    ax = plt.subplot(gs1[0], projection='3d')
    ax.view_init(18, -70) 

    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    
    #train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
    #actions, FLAGS.data_dir)
    
    data_mean_2d = np.array([534.23620678, 425.88147731, 533.39373118, 425.69835782, 534.39780234,
    497.43802989, 532.48129382, 569.04662344, 0, 0,
    0, 0, 535.29156558, 425.73629462, 532.76756177,
    496.47315471, 530.88803412, 569.60750683, 0, 0,
    0, 0, 0, 0, 535.75344606, 331.22677323, 536.33800053, 317.44992858, 0, 0, 536.71629464,
    269.11969467, 0, 0, 536.36740264, 330.27798906, 535.85669709, 374.59944401, 534.70288482, 387.35266055,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 534.99202566, 331.77346527, 534.61130808, 373.81679139, 535.21529192,
    379.50779675, 0, 0, 0, 0, 0, 0, 0, 0])

    data_std_2d =  np.array([107.37361012, 61.63490959, 114.89497053,  61.91199702, 117.27565511, 
    50.22711629, 118.89857526, 55.00005709, 0, 0,  0, 0, 102.91089763, 61.33852088, 106.66944715,
    47.96084756, 108.13481259, 53.60647266, 0, 0,  0, 0, 0, 0, 110.56227428,
    72.9166997, 110.84492972, 76.09916643, 0, 0, 115.08215261, 82.92943734, 0, 0, 105.04274864,
    72.84070269, 106.3158104, 73.21929021, 108.08767528, 82.4948776, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 124.31763034, 73.34214366, 132.80917569, 76.37108859, 131.60933137,
    88.82878858, 0, 0, 0, 0, 0, 0, 0, 0])

    dim_to_use_2d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 30, 31, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55])


    #train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    #actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)#2d joints
    
        print(humans)

        if len(humans) > 0:
            ret = converd_skeleton_human(humans[0].get_parts(),image.shape[1],image.shape[0])
            _, _, poses3d = op3dgraph.step(session3d, ret[0], ret[1], ret[2], isTraining=False)#3d skeleton
            viz.show3Dpose(poses3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")
            

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)#draw

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
