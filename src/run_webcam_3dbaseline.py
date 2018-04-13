import argparse
import logging
import time

import cv2
import numpy as np
import sys
sys.path.append("C:/Users/laura/Downloads/tf-pose-estimation/src/baseline3d")
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import tensorflow as tf
from baseline3d.predict_3dpose import create_model
import data_utils
import viz
import cameras

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
dim_to_ignore_2d=np.array([8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 28, 29 ,32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 56, 57, 58, 59, 60, 61, 62, 63])
dim_to_ignore_3d=np.array([0, 1, 2, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49, 50, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])
data_mean_3d=np.array([0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.75354470e+01, -1.36709039e+00, 4.92844242e+00, -7.90273729e+01, -1.20514378e+02, -3.28446097e+02, -5.79440771e+01, -6.73929398e+01, -7.10121376e+02, -5.34517822e+01, -1.57303491e+02, -7.61004578e+02, -5.94314764e+01,
 -2.01113859e+02, -7.54981241e+02, 8.75349517e+01, 1.36710863e+00, -4.92839771e+00, 9.39525086e+01, -1.22135208e+02, -3.31073411e+02, 7.95681993e+01, -7.99901279e+01, -7.23553599e+02, 8.73690011e+01,
 -1.73848463e+02, -7.63215149e+02, 9.37853969e+01, -2.17441040e+02, -7.49781389e+02, 2.23585794e-03, 3.95137219e-03, 8.82909486e-02, -3.94043478e+00, 1.11653861e+01, 2.16721752e+02, -9.16574833e+00,
 -1.87125289e+01, 4.46726844e+02, -1.38790486e+01, -7.65703525e+01, 5.05840752e+02, -2.25257442e+01, -4.88483976e+01, 5.99628081e+02, -9.16574833e+00, -1.87125289e+01, 4.46726844e+02, 8.03854542e+01,
 -1.08273287e+01, 3.96518642e+02, 1.65196695e+02, -2.93027070e+01, 1.97611881e+02, 1.24325826e+02, -1.01367920e+02, 1.28396193e+02, 1.24325826e+02, -1.01367920e+02, 1.28396193e+02, 1.01086119e+02,
 -1.06778225e+02, 1.60298441e+02, 1.29438790e+02, -1.30262978e+02, 1.16278538e+02, 1.29438790e+02, -1.30262978e+02, 1.16278538e+02, -9.16574833e+00, -1.87125289e+01, 4.46726844e+02, -9.60294095e+01,
 -1.25862897e+01, 3.91452494e+02, -1.69875754e+02, -3.72685042e+01, 1.98147185e+02, -1.29945625e+02, -1.11232009e+02, 1.71318467e+02, -1.29945625e+02, -1.11232009e+02, 1.71318467e+02, -1.15184617e+02,
 -1.09535229e+02, 2.04763247e+02, -1.35368678e+02, -1.46341162e+02, 1.68847216e+02, -1.35368678e+02, -1.46341162e+02, 1.68847216e+02])
data_std_3d=np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.84781829e+01, 6.26004666e+01, 1.72345035e+01, 1.12010977e+02, 1.86900452e+02, 1.89314773e+02, 1.50607808e+02, 2.53630374e+02, 2.44869495e+02, 1.70427150e+02, 2.79880763e+02, 2.63539191e+02, 1.86283220e+02,
 2.95439365e+02, 2.71854934e+02, 7.84777295e+01, 6.26001005e+01, 1.72344196e+01, 1.15640855e+02, 1.90546053e+02, 2.00973056e+02, 1.51240145e+02, 2.46933449e+02, 2.52103733e+02, 1.76316153e+02, 2.72401336e+02, 2.67217756e+02, 1.93739068e+02, 2.88385685e+02,
 2.74136354e+02, 1.65501143e-02, 2.84645651e-02, 3.31657017e-02, 4.57644849e+01, 7.32142160e+01, 5.07131046e+01, 8.32551061e+01, 1.34857550e+02, 9.86138797e+01, 1.02871840e+02, 1.53485788e+02, 1.22743200e+02, 1.10739959e+02, 1.63330752e+02, 1.27573080e+02,
 8.32551061e+01, 1.34857550e+02, 9.86138797e+01, 1.10207643e+02, 1.43404241e+02, 9.31017009e+01, 1.81133544e+02, 1.96195489e+02, 1.25050366e+02, 2.08126338e+02, 2.17639290e+02, 2.13888261e+02, 2.08126338e+02, 2.17639290e+02, 2.13888261e+02, 2.00800713e+02,
 2.10760023e+02, 2.22070893e+02, 2.44192285e+02, 2.50397388e+02, 2.56138265e+02, 2.44192285e+02, 2.50397388e+02, 2.56138265e+02, 8.32551061e+01, 1.34857550e+02, 9.86138797e+01, 1.10526814e+02, 1.40825054e+02, 9.78640867e+01, 1.80824856e+02, 1.99868127e+02,
 1.40180146e+02, 2.08748406e+02, 2.29725741e+02, 2.38826045e+02, 2.08748406e+02, 2.29725741e+02, 2.38826045e+02, 2.04270964e+02, 2.22062718e+02, 2.35910316e+02, 2.52303429e+02, 2.73158817e+02, 3.05599841e+02, 2.52303429e+02, 2.73158817e+02, 3.05599841e+02])
    
def convert_skeleton_human(xy,image_w,image_h):
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
    return (enc_in,dec_out,dp,spine_x,spine_y)
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

    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    session3d = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1},
           allow_soft_placement=True))
    batch_size = 128
    op3dgraph = create_model(session3d, batch_size)
    
        ###FOR PLOTING
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    ax = plt.subplot(gs1[0], projection='3d')
    ax.view_init(18, -70)

    ### Camera reading###

    #logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    nf=0;
    while True:
        ret_val, image = cam.read()
        #logger.debug('image preprocess+')
        ##Image scaling
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

        #logger.debug('image process+')
        ## 2D Estimation
        humans = e.inference(image)#2d joints
        print(len(humans))
        print('LENGTH')
        ##3D Estimation
        if len(humans) > 0:
            ret = convert_skeleton_human(humans[0].get_parts(),image.shape[1],image.shape[0])
            _, _, poses3d = op3dgraph.step(session3d, ret[0], ret[1], ret[2], isTraining=False)#3d skeleton
            ##Unnormalization of the data
            all_poses_3d = []
            #enc_in = data_utils.unNormalizeData(ret[0], data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            
            #print(poses3d)
            
            all_poses_3d.append( poses3d )
            poses3d = np.vstack(all_poses_3d)
            subplot_idx, exidx = 1, 1
            max = 0
            min = 10000

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    tmp = poses3d[i][j * 3 + 2]
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp
                    if poses3d[i][j * 3 + 2] > max:
                        max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < min:
                        min = poses3d[i][j * 3 + 2]

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = max - poses3d[i][j * 3 + 2] + min
                    poses3d[i][j * 3] += (ret[3] - image.shape[1])#630=image.shape[1]
                    poses3d[i][j * 3 + 2] += (image.shape[0] - ret[4])#500=image.shape[0]

            

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)#draw

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')
        if len(humans) > 0 and nf>0:
            #print(poses3d)
            #print('Sim é verdade!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # Plot 3d predictions
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
            plt.axis('off')
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)
            #logger.debug(np.min(poses3d))
            if np.min(poses3d) < -1000:
                poses3d = before_pose

            #p3d = poses3d
            #logger.debug(poses3d)
            viz.show3Dpose(poses3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")
            before_pose = poses3d
            '''
            pngName = 'png/test_{0}.png'.format(str(frame))
            plt.savefig(pngName)
            png_lib.append(imageio.imread(pngName))
            '''
            
        nf=1
            
        

    cv2.destroyAllWindows()
    session3d.close()
    '''

    sess.close()
    
    '''
