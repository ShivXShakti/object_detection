import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point

import numpy as np
from scipy.linalg import logm, expm
import time
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation as R
from custom_interface.srv import ObjectDetection 

class TrajectoryGenerator(Node):
    """
    output: x, x_dot
    """ 
    def __init__(self, T_init = None, T_final=None, traj_time=10.0, sampling_frequency=100):
        super().__init__('trajectory_generator_node')
        self.cli = self.create_client(ObjectDetection, 'detect_object')
        print("started init")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        print("after service")

        self.req = ObjectDetection.Request()
        
        self.publisher_pose = self.create_publisher(Float64MultiArray, "/ur_trajectory_generator/pose_ref", 10)
        self.publisher_posedot = self.create_publisher(Float64MultiArray, "/ur_trajectory_generator/posedot_ref", 10)
        
        '''self.subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_positions',
            self.joint_command_callback,
            10)'''
        """self.subscription = self.create_subscription(
            Point,
            '/target_pose',
            self.listener_callback,
            10
        )"""
        #self.subscription  # prevent unused variable warning
        self.pf = np.array([0,0,0])
        self.once = True
        self.received_pose = False

        
        self.sampling_frequency = sampling_frequency
        # Timer to publish at 125Hz (every 8ms)
        self.timer = self.create_timer(1/self.sampling_frequency, self.publish_trajectory)
        
        self.traj_time = traj_time
        l1,l2, l3, l4, l5, l6, l7, l8 = np.array([0.1273, 0.220491, 0.612, 0.1719, 0.5723, 0.1149, 0.1157, 0.0922])

        self.T_init = np.array([[0, 1, 0, l5+l7], 
                                 [1, 0, 0, l2-l4+l6], 
                                 [0, 0, -1, l1+l3-l8], 
                                 [0, 0, 0, 1]]) if T_init is None else T_init
        
        
        
        ### gate face
        '''self.T_final = np.array([[0, 1, 0, l5+l7],    
                                 [0, 0, 1, l2-l4+l6+l8], 
                                 [1, 0, 0, l1+l3], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final'''
        
        ## eer
        '''self.T_final = np.array([[-1, 0, 0, l5+l7], 
                                 [0, 1, 0, l2-l4+l6], 
                                 [0, 0, -1, l1+l3-l8], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final'''
        ## pcface
        '''self.T_final = np.array([[1, 0, 0, l2-l4+l6], 
                                 [0, -1, 0, -l5-l7], 
                                 [0, 0, -1, l1+l3-l8], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final'''
        ## elegant door looking
        self.T_final = np.array([[0, 1, 0, self.pf[0]], 
                                 [1, 0, 0, self.pf[1]], 
                                 [0, 0, -1, 0.4], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final
        
        '''self.T_init = np.eye(4) if T_init is None else T_init
        Rx = np.array([[1, 0, 0],
               [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
               [0, np.sin(np.radians(90)), np.cos(np.radians(90))]])
        
        Ry = np.array([[np.cos(np.radians(90)), 0, np.sin(np.radians(90))],
               [0, 1, 0],
               [-np.sin(np.radians(90)), 0, np.cos(np.radians(90))]])

        Rz = np.array([[np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
               [np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
               [0, 0, 1]])
        self.r = Rx @ np.eye(3)
        T_final = np.eye(4)
        T_final[:3,:3] = self.r
        T_final[:3,3] = np.array([1,1,1])
        self.T_final = T_final'''
        '''self.T_final = np.array([[0, 1, 0, 0.5723+0.1157+0.5], 
                                 [1, 0, 0, 0.220491-0.1719+0.1149], 
                                 [0, 0, -1, 0.1273+0.612-0.0922], 
                                 [0, 0, 0, 1]]) if T_final is None else T_final'''
        
        
        self.time_counter = 0.0
        
        self.prev_position = None
        self.prev_euler = None
        self.plot_pose = []
        self.detection_resp = False
        #self.latest_positions = []
        
    
    def send_request(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.detection_resp = response.success
            print(f"response: {self.detection_resp}")
            if self.detection_resp:
                self.get_logger().info(f"Detected {len(response.objects)} objects:")
                self.start_time = time.time()
                for obj in response.objects:
                    self.get_logger().info(f"Label: {obj.label}, x: {obj.x}, y: {obj.y}, z: {obj.z}")
                    self.pf = np.array([obj.x, -obj.y, obj.z])
                    self.T_final = np.array([[0, 1, 0, self.pf[0]], 
                                            [1, 0, 0, self.pf[1]], 
                                            [0, 0, -1, 0.4], 
                                            [0, 0, 0, 1]])
            else:
                self.get_logger().info("Detection failed or no objects found.")
        else:
            self.get_logger().error('Service call failed')

    
    def rot2eul(self,rot_matrix, seq='XYZ'):
        """
        Convert rotation matrix to Euler angles.
        Uses scipy's spatial transform for high efficiency.
        """
        rotation = R.from_matrix(rot_matrix)
        eul = rotation.as_euler(seq, degrees=False)
        return eul
        
    def eul2angular_vel(self,euler):
        alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot = euler

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        omega_x = ((ca * sg + cg * sa * sb) * 
                   (ca * sg * alpha_dot + cg * sa * gamma_dot - ca * cb * cg * beta_dot + 
                    cg * sa * sb * alpha_dot + ca * sb * sg * gamma_dot) +(ca * cg - sa * sb * sg) * 
                    (ca * cg * alpha_dot - sa * sg * gamma_dot + ca * cb * sg * beta_dot +
                    ca * cg * sb * gamma_dot - sa * sb * sg * alpha_dot) +
                    cb * sa * (cb * sa * alpha_dot + ca * sb * beta_dot))

        omega_y = ca * beta_dot - cb * sa * gamma_dot
        omega_z = sa * beta_dot + ca * cb * gamma_dot

        return np.array([omega_x, omega_y, omega_z])

    def compute_differentiation(self, prev_pose, curr_pose, dt):
        """
        Compute linear velocity given previous pose, current pose, and time increment.

        Args:
            prev_pose (np.array): Previous position (x, y, z)
            curr_pose (np.array): Current position (x, y, z)
            dt (float): Time increment (should be > 0)

        Returns:
            np.array: Linear velocity (vx, vy, vz)
        """
        if dt <= 0:
            raise ValueError("Time increment must be positive.")
    
        velocity = (curr_pose - prev_pose) / dt
        return velocity
    
    def trajectory_smooth(self):
        """ Generate smooth trajectory using the exponential matrix method. """

        R_init = self.T_init[:3, :3]
        R_final = self.T_final[:3, :3]
        position_init = self.T_init[:3, 3]
        position_final = self.T_final[:3, 3]

        # Compute interpolated rotation using exponential map
        R_instant = R_init @ expm(logm(R_init.T @ R_final) * self.time_counter/self.traj_time)
        position_instant = position_init + self.time_counter/self.traj_time * (position_final - position_init)
        euler_instant = self.rot2eul(R_instant)

        if self.prev_position is None:
            self.prev_position = position_init
        position_dot = self.compute_differentiation(self.prev_position, position_instant, 1/self.sampling_frequency)
        
        if self.prev_euler is None:
            self.prev_euler = self.rot2eul(R_init)
        euler_dot = self.compute_differentiation(self.prev_euler, euler_instant, 1/self.sampling_frequency)
        orientation_dot = self.eul2angular_vel(np.hstack([euler_instant, euler_dot]))
        self.prev_position, self.prev_euler = position_instant, euler_instant
        self.time_counter += (1/self.sampling_frequency)
        return np.hstack([euler_instant, position_instant]), np.hstack([orientation_dot, position_dot])

    def publish_trajectory(self):
        if self.detection_resp == False:
            return
        elapsed_time = time.time() - self.start_time
        #if elapsed_time > self.traj_time:  # Stop after 5 seconds
        #    if False:
         #       self.plot(self.plot_pose)
         #       self.plot3d(self.plot_pose)
            #self.get_logger().info("5 seconds elapsed, stopping publisher and shutting down node.")
            #self.timer.cancel()
            #self.destroy_node()  
            #rclpy.shutdown()  # Shutdown ROS2
        if elapsed_time<=self.traj_time:
            pose, pose_dot = self.trajectory_smooth()
            self.plot_pose.append(pose)
            if not pose.any():
                print("No trajectory point received")
                return
            msg_pose = Float64MultiArray()
            msg_pose.data = [i for i in pose]  # List of six float values
            msg_pose_dot = Float64MultiArray()
            msg_pose_dot.data = [i for i in pose_dot]
        
            self.publisher_pose.publish(msg_pose)
            self.publisher_posedot.publish(msg_pose_dot)
            self.get_logger().info(f'Pose: {msg_pose.data}, Posedot: {msg_pose_dot.data}')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    node.send_request()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
