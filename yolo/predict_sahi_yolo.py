

import time

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path

weights = '/home/maantonov_1/VKR/actual_scripts/yolo/runs/detect/yolov8s/weights/best.pt'

yolov8_model_path = f"{weights}"
download_yolov8s_model(yolov8_model_path)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
)


frame = cv2.imread(f'/home/maantonov_1/VKR/data/small_train/test/images/1.jpg', cv2.IMREAD_COLOR)
# frame = '/home/maantonov_1/VKR/data/small_train/test/images/1.jpg'

st = time.time()
results = get_sliced_prediction(
    frame, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2
)
st = time.time() - st
print(f'time: {st}', flush = True)
object_prediction_list = results.object_prediction_list

boxes_list = []
clss_list = []
for ind, _ in enumerate(object_prediction_list):
    boxes = (
        object_prediction_list[ind].bbox.minx,
        object_prediction_list[ind].bbox.miny,
        object_prediction_list[ind].bbox.maxx,
        object_prediction_list[ind].bbox.maxy,
    )
    clss = object_prediction_list[ind].category.name
    boxes_list.append(boxes)
    clss_list.append(clss)

for box, cls in zip(boxes_list, clss_list):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
    label = str(cls)
    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
    cv2.rectangle(
        frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
    )
    cv2.putText(
        frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
    )

cv2.imwrite(f'/home/maantonov_1/VKR/actual_scripts/yolo/1.jpg', frame)



