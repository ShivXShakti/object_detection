import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from geometry_msgs.msg import Point


class YoloRealsenseNode(Node):
    def __init__(self):
        super().__init__('yolo_realsense_node')
        self.subscription = self.create_subscription(
            Image,
            '/robot1/D435_1/color/image_raw',
            self.image_callback,
            10)
        self.create_subscription(Image, '/robot1/D435_1/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/robot1/D435_1/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)

        self.timer = self.create_timer(1.0, self.driver)
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("YOLO Realsense Node Initialized")
        self.depth_image = None
        self.color_image = None
        self.publisher_ = self.create_publisher(Point, '/target_pose', 10)

    def camera_info_callback(self, msg):
        """Extract intrinsic camera parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
    def depth_callback(self, msg):
        """Convert depth image from ROS2 message to OpenCV format"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
    def image_callback(self, msg):
        #self.get_logger().info("Received image from camera")
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        
    def depth_2_cartisian(self, depth_img, color_img):
        if depth_img is None or self.camera_matrix is None:
            self.get_logger().warn("Waiting for depth image and camera info...")
            return
        results = self.model(color_img)
        for result in results:
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = obj.tolist()
                class_name = self.model.names[int(class_id)]
                
                cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_img, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                """cv2.rectangle(depth_img, (int(cx), int(cy)), (int(cx+1.0), int(cy+1.0)), (0, 255, 0), 5)
                # Normalize for visualization
                depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = np.uint8(depth_normalized)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)"""

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
                print(f"wold cordinate: {p.flatten(), p[0]}")
                p = p.flatten()
                point = Point()
                point.x = p[0]
                point.y = p[1]
                point.z = p[2]
                self.publisher_.publish(point)

                self.get_logger().info(f'Published coordinates: x={X}, y={Y}, z={Z}')
        
            cv2.imshow("YOLO Detection", color_img)
            #cv2.imshow("Depth window", depth_colormap)
            cv2.waitKey(1)
    def driver(self):
        self.depth_2_cartisian(depth_img=self.depth_image, color_img=self.color_image)

    
        
        
def main(args=None):
    rclpy.init(args=args)
    node = YoloRealsenseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



