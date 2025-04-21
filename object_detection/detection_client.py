import rclpy
from rclpy.node import Node
from custom_interface.srv import ObjectDetection  # Your custom service

class ObjectDetectionClient(Node):
    def __init__(self):
        super().__init__('object_detection_client')
        self.cli = self.create_client(ObjectDetection, 'detect_object')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.req = ObjectDetection.Request()

    def send_request(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Detected {len(response.objects)} objects:")
                for obj in response.objects:
                    self.get_logger().info(f"Label: {obj.label}, x: {obj.x}, y: {obj.y}, z: {obj.z}")
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
