# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils import contours
import numpy as np
import time
import cv2
import torch
from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1_out, CrowdCounter_cnterr_LP, CrowdCounter_cnterr_LA
from src import network
import math


def calculate_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/mall/MCNN-final-1/MCNN-final-1_mall_6_crop_50.h5'
#model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver3/MCNN-ver3_schtechA_35_crop_50.h5'
#model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver2/MCNN-ver2_schtechA_54_crop_50.h5'
#input_video_path = 'C:/Users/jalee/Desktop/FYP/test/video/output.mp4'

input_video_path = 'C:/Users/jalee/Desktop/FYP/Test1.mp4'

vs = FileVideoStream(input_video_path).start()
# vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

net = CrowdCounter_cnterr_LP()
net.cuda()
network.load_net(model_path, net)

frame_num = 0
clear_count = 0
array_init = []
array_final = []
path_list = []
new_pts_added = []

min_dist = 8
max_dist = 30

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (640, 480))  # 720p - 1280 x 720
    unedited_frame = frame.copy()
    (h, w) = frame.shape[:2]
    if frame_num % 10 == 0:
        clear_count = clear_count + 1
        blob = cv2.dnn.blobFromImage(frame)
        blob = torch.tensor(blob)
        blob = blob.cuda()

        density_map = net(blob)

        density_map = density_map.data.cpu().numpy()

        density_map = density_map[0][0]
        density_map = 255 * density_map / np.max(density_map)

        # transfer to 3 channels
        density_map = torch.from_numpy(density_map)
        density_map.unsqueeze_(2)
        density_map = density_map.repeat(1, 1, 3)
        density_map = density_map.numpy()
        density_map = density_map.astype(np.float32, copy=False)

        if density_map.shape[1] != frame.shape[1]:
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))

        density_map = density_map.astype(np.uint8)

        thresh = 100
        max_val = 250
        # cv2.imshow('img before threshold', img)
        ret, img2 = cv2.threshold(density_map, thresh, max_val, cv2.THRESH_BINARY)

        img2 = cv2.erode(img2, None, iterations=2)
        img2 = cv2.dilate(img2, None, iterations=2)

        imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        # cnts = contours.sort_contours(cnts)[0]

        # Getting coordinates of people
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            # cv2.circle(frame, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)
            if frame_num == 0:
                array_init.append([int(cX), int(cY)])
            else:
                array_final.append([int(cX), int(cY)])

        num_people = len(cnts)

        # print('init array at start')
        # print(array_init)
        # print('final array at start')
        # print(array_final)

        if frame_num == 0:
            for i, n in enumerate(array_init):
                path_list.append([])
                path_list[i].append(n)

        temp_array_final = array_final.copy()

        if frame_num != 0:
            for coord_init in array_init:
                x_init = coord_init[0]
                y_init = coord_init[1]
                dist = max_dist
                # print('')
                valid_dist_flag = False
                for coord_final in temp_array_final:
                    # print(temp_array_final)
                    x_final = coord_final[0]
                    y_final = coord_final[1]
                    # print('1')
                    # print('\nx init: %d \ty init: %d' % (x_init, y_init))
                    # print('x final: %d \ty final: %d' % (x_final, y_final))
                    # print('dist: %d' % calculate_distance(x_final, y_final, x_init, y_init))
                    # print('initial dist: %d' % dist)
                    if calculate_distance(x_final, y_final, x_init, y_init) < min_dist:
                        x_path = x_final
                        y_path = y_final
                        valid_dist_flag = True
                        # print('dist = 0')
                        # print('same pts')
                        break

                    elif calculate_distance(x_final, y_final, x_init, y_init) < dist:
                        valid_dist_flag = True
                        dist = calculate_distance(x_final, y_final, x_init, y_init)
                        x_path = x_final
                        y_path = y_final
                        # print('dist != 0 and < max dist')

                    # elif valid_dist_flag is False and calculate_distance(x_final, y_final, x_init, y_init) >= max_dist:
                        # x_path = x_final
                        # y_path = y_final
                        # print('dist > max dist')

                # print('flag')
                # print(valid_dist_flag)
                # print('x init: %d \ty init: %d' % (x_init, y_init))
                # print('x path: %d \ty path: %d' % (x_path, y_path))
                # print('dist: %d' % calculate_distance(x_path, y_path, x_init, y_init))

                if valid_dist_flag is True and calculate_distance(x_path, y_path, x_init, y_init) < max_dist:
                    # print('temp array before removal')
                    # print(temp_array_final)
                    temp_array_final.remove([x_path, y_path])
                    # print('temp array after removal')
                    # print(temp_array_final)
                # else:
                    # print('no element removed')

                if calculate_distance(x_path, y_path, x_init, y_init) == dist and calculate_distance(x_path, y_path, x_init, y_init) > min_dist:
                    for i, n in enumerate(path_list):
                        if x_init == n[-1][0] and y_init == n[-1][1]:
                            n.append([x_path, y_path])
                            # print(i)
                            # print('use already available pt\n')
                            break
                        if i == len(path_list) - 1:
                            path_list.append([[x_path, y_path]])
                            # print('\nuse new pt')
                            break
                    # print('dist > threshold dist')
                new_pts_added.append([x_path, y_path])
                # print('path list')
                # print(path_list)

            array_init = array_final.copy()
            array_final.clear()

        # update path list every 10 frames
        if frame_num % 10 == 0 and frame_num != 0:
            # restore frame as image with no drawn line
            frame = unedited_frame
            keep = False
            for paths in path_list:
                for pts in paths:
                    for n in new_pts_added:
                        if pts[0] == n[0] and pts[1] == n[1]:
                            keep = True
                if keep is not True:
                    path_list.remove(paths)
                    keep = True
        new_pts_added = []

        # print(array_init)
        # print(array_final)
        color = (0, 0, 255)
        # draw lines of paths
        for paths in path_list:
            b, g, r = color
            b = b + 10
            g = g + 10
            r = r - 10
            color = (b, g, r)
            if len(paths) != 1:
                # print('\npaths')
                # print(paths)
                for i in range(1, len(paths)):
                    # cv2.line(frame, (paths[i][0], paths[i][1]), (paths[i - 1][0], paths[i - 1][1]), color, 2)
                    cv2.arrowedLine(frame, (paths[i - 1][0], paths[i - 1][1]), (paths[i][0], paths[i][1]), color, 2)


        result_img = np.hstack((frame, img2))

        # show the output frame
        cv2.imshow("Output2", frame)
        # show the output frame
        # cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.waitKey(0)
        # update the FPS counter
        fps.update()

    frame_num = frame_num + 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
