
### https://chiachun0818.medium.com/?p=248e369b93c3

import cv2
import darknet
import time

# Parameters
win_title = 'Yolov4 customer detector'
cfg_file = 'cfg/yolov4-tiny.cfg'
data_file= 'cfg/coco.data'
weight_file = 'yolov4-tiny.weights'

thre = 0.15
show_coordinates = True


network, class_names, class_colors = darknet.load_network(cfg_file, data_file, weight_file, batch_size=1)
width = darknet.network_width(network)
height = darknet.network_height(network)


cap = cv2.VideoCapture('/dev/video0')


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_prev = time.time()

    #   fix image format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))

    #   convert to darknet image format
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    #   inference
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)

    fps = int(1/(time.time()-t_prev))
    print("fps: ", fps)

    #   draw 
    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #   show image and FPS
    cv2.rectangle(image, (5, 5), (72, 25), (0, 0, 0), -1)
    cv2.putText(image, f'FPG {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(win_title, image)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destoryAllWindows()
cap.release()


