import rclpy
from rclpy.node import Node
from custom_interface.srv import ObjectDetection
from custom_interface.msg import ObjectArray

class ObjectDetectionClient(Node):
    def __init__(self):
        super().__init__('object_detection_client')
        self.cli = self.create_client(ObjectDetection, 'detect_object')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.req = ObjectDetection.Request()
        self.publisher_ = self.create_publisher(ObjectArray, '/detected_objects', 10)

    def send_request(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                object_array_msg = ObjectArray()
                object_array_msg.objects = response.objects
                self.publisher_.publish(object_array_msg)
                self.get_logger().info(f"Detected Objects: {response.objects}")
            else:
                self.get_logger().info("Detection failed or no objects found.")
        else:
            self.get_logger().error('Service call failed')

def main():
    rclpy.init()
    client = ObjectDetectionClient()
    client.send_request()
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
