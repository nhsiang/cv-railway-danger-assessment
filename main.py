import cv2 as cv
import numpy as np
from ultralytics import YOLO

# model_detection = YOLO("models/yolo26n.pt")
model_segmentation = YOLO("models/runs/semantic/train/weights/best.pt")
video = cv.VideoCapture('./assets/video/kibukawa_to_minakuchijyonan.mp4')

while True:
    isTrue, frame = video.read()
    frame = frame[540:1060,580:1450]

    results_segmentation = model_segmentation(frame)
    masks = results_segmentation[0].semantic_mask.data.cpu().numpy()

    # for i in [1, 2, 3]: # verifying masks
    #     mask = (masks == i).astype(np.uint8) * 255
    #     cv.imshow(f"class {i}", mask)

    track_mask = (masks == 1).astype(np.uint8)
    
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
    
    # output
    screen = frame.copy()
    cv.putText(screen, f'median curvature: {k_frame}', (50, 50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
    
    cv.imshow("YOLO", track_mask)
    cv.imshow("output", screen)
    # cv.imshow("output", results_segmentation[0].plot())

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()