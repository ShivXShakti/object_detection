import rclpy
from rclpy.node import Node
from custom_interface.srv import ObjectDetection
from custom_interface.msg import Object
from sensor_msgs.msg import Image, CameraInfo
import cv2
from ultralytics import YOLO
import numpy as np
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class ObjectDetectionServer(Node):
    def __init__(self):
        super().__init__('object_detection_server')
        self.srv = self.create_service(ObjectDetection, 'detect_object', self.depth_2_cartisian)
        self.get_logger().info('Object Detection Service Ready')

        self.subscription = self.create_subscription(
            Image,
            '/robot1/D435_1/color/image_raw',
            self.image_callback,
            10)
        self.create_subscription(Image, '/robot1/D435_1/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/robot1/D435_1/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("YOLO Realsense Node Initialized")
        self.depth_image = None
        self.color_image = None

    
    def camera_info_callback(self, msg):
        """Extract intrinsic camera parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
    def depth_callback(self, msg):
        """Convert depth image from ROS2 message to OpenCV format"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
    def image_callback(self, msg):
        #self.get_logger().info("Received image from camera")
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        
    def depth_2_cartisian(self, request, response):
        """Service callback to detect multiple objects"""
        depth_img, color_img = self.depth_image, self.color_image
        if color_img is None and depth_img is None:
            response.success = False
            self.get_logger().warn("No image received yet.")
            return response
        results = self.model(color_img)
        #detected_objs = []
        #labels = []
        detected_objects = []
        for result in results:
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = obj.tolist()
                class_name = self.model.names[int(class_id)]
                
                cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_img, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                Z = depth_img[cy, cx] * 0.001  # Depth in mm
                if Z == 0:  # Ignore invalid depth points
                    self.get_logger().warn("Invalid depth at selected point!")
                    return

                # Convert to 3D coordinates
                X = (cx - self.camera_matrix[0, 2]) * Z / self.camera_matrix[0, 0]
                Y = (cy - self.camera_matrix[1, 2]) * Z / self.camera_matrix[1, 1]

                TWC = np.array([[1,0,0, 0.400],
                        [0,0,-1, 1.000],
                        [0,-1,0,0.110],
                        [0,0,0,1]])
                poc = np.array([X, Y, Z, 1]).reshape(4,1)
                p = TWC@poc
                #labels.append(class_name)
                #detected_objs.append(p.flatten())
                p = p.flatten()
                obj = Object()
                obj.label = class_name
                obj.x = p[0]
                obj.y = p[1]
                obj.z = p[2]
                print(f"wold cordinate: {p.flatten(), p[0]}")
                self.get_logger().info(f'Published coordinates: x={X}, y={Y}, z={Z}')

                detected_objects.append(obj)
                #cv2.imshow("YOLO Detection", color_img)
                #cv2.imshow("Depth window", depth_colormap)
                #cv2.waitKey(1)
        response.objects = detected_objects
        response.success = True if detected_objects else False
        self.get_logger().info(f"Detected {len(detected_objects)} objects.")
        return response
                

def main():
    rclpy.init()
    node = ObjectDetectionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
