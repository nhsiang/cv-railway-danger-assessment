import cv2 as cv
import numpy as np
from ultralytics import YOLO

model_detection = YOLO("models/yolo26n.pt")
model_segmentation = YOLO("models/runs/semantic/train/weights/best.pt")
video = cv.VideoCapture('./assets/video/curved_track.mp4')

while True:
    isTrue, frame = video.read()
    is_overlapping = False

    results_detection, results_segmentation = model_detection(frame), model_segmentation(frame)
    boxes = results_detection[0].boxes.xyxy.cpu().numpy().astype(int)
    masks = results_segmentation[0].semantic_mask.data.cpu().numpy()

    # for i in [1, 2, 3]: # verifying masks
    #     mask = (masks == i).astype(np.uint8) * 255
    #     cv.imshow(f"class {i}", mask)

    track_mask = (masks == 1).astype(np.uint8)

    # check object bounding box overlap with rail properties
    for x1, y1, x2, y2 in boxes:
        box_region = masks[y1:y2, x1:x2]
        if np.any(np.isin(box_region, [1, 2, 3])):
            is_overlapping = True
            break

    # plot railway path (based on mask rail-track)
    coords = []
    for y in range(track_mask.shape[0] - 1, 0, -10):
        x = np.where(track_mask[y] > 0)[0]
        if len(x) > 0:
            left = x[0]
            right = x[-1]
            mid_x = (left + right) // 2
            coords.append((mid_x, y))
    coords = np.array(coords)

    # calculate curvature
    f = np.polyfit(coords[:, 1], coords[:, 0], 2)  # x=f(y)=ay^2+by+c
    f_p, f_pp = np.polyder(f, 1), np.polyder(f, 2)
    y = coords[:, 1]
    dx, ddx = np.polyval(f_p, y), np.polyval(f_pp, y)
    k = np.abs(ddx) / (1 + dx ** 2) ** 1.5  # κ(y)=|f''(y)|/(1+[f'(y)]^2)^(3/2)
    k_frame = np.median(k)

    # verdict
    curve_threshold = 0.001
    if k_frame > curve_threshold:
        status = "CURVE AHEAD"
    else:
        status = "NORMAL"

    if is_overlapping:
        detect_status = "CAUTION: OBJECT ON RAIL"
    else:
        detect_status = "NORMAL"

    # output
    screen = results_detection[0].plot()
    cv.putText(screen, f'Relative curvature: {k_frame}', (50, 50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
    cv.putText(screen, status, (50, 100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
    cv.putText(screen, detect_status, (50, frame.shape[0]-50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)

    # ys = np.arange(coords[:, 1].min(), coords[:, 1].max()) # verify plotted line
    # pts = np.array([(np.polyval(f, y), y) for y in ys], dtype=np.int32)
    # cv.polylines(screen, [pts], False, (255, 0, 0), 2)

    cv.imshow("output", screen)
    # cv.imshow("output", results_segmentation[0].plot())

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()