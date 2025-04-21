import torch
import cv2
from ultralytics import YOLO
import signal
import sys


class ObjectDetection:
    def __init__(self, model_name = "yolov8n.pt"): # Use "yolov8s.pt" for better accuracy
        self.model_name = model_name
        self.model = YOLO(model_name)
    def detect(self, image_path = "/home/cstar/Documents/urs_ws/src/ur_ai/images/bottle.jpeg"):
        #image = cv2.imread(image_path)
        results = self.model(image_path)
        obj_labels = []
        obj_bb = []
        for result in results:
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = obj.tolist()
                class_name = self.model.names[int(class_id)]
                obj_bb.append([x1, y1, x2, y2])
                obj_labels.append(class_name)
                print(f"detected class: {class_name}, {[x1, y1, x2, y2]}")
                cv2.rectangle(image_path, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image_path, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Detected Objects", image_path)
            cv2.waitKey(10000)
            #cv2.destroyAllWindows()
        """if img_show:
            #cv2.imwrite("output.jpg", image)
            cv2.imshow("Detected Objects", image)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()"""
        return obj_labels, obj_bb
if __name__=="__main__":
    instance = ObjectDetection()
    instance.detect()