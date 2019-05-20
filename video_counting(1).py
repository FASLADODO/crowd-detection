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

#model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver3/MCNN-ver3_schtechA_35_crop_50.h5'
#model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver2/MCNN-ver2_schtechA_54_crop_50.h5'
model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/mall/MCNN-final-1/MCNN-final-1_mall_6_crop_50.h5'
#input_video_path = 'C:/Users/jalee/Desktop/FYP/test/video/output.mp4'
input_video_path = 'C:/Users/jalee/Desktop/FYP/Test1.mp4'

vs = FileVideoStream(input_video_path).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

net = CrowdCounter_cnterr_LP()
net.cuda()
network.load_net(model_path, net)


while True:
    frame = vs.read()
    frame = cv2.resize(frame, (640, 480))  # 720p - 1280 x 720
    (h, w) = frame.shape[:2]

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
    cnts = contours.sort_contours(cnts)[0]

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(frame, (int(cX), int(cY)), int(radius), (0, 0, 255), 2)

    num_people = len(cnts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_text = 0  # img3.shape[0] - 200
    y_text = 30  # img3.shape[1] - 200
    cv2.putText(frame, 'Num people detected: %d' % num_people, (x_text, y_text), font, 1, (255, 255, 255))

    # result_img = np.hstack((frame, img2))

    # show the output frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord(" "):
        cv2.waitKey(0)

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
