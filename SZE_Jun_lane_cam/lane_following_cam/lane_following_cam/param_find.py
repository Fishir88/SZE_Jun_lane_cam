import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class Paramfind(Node):
    def __init__(self):
        super().__init__('parameter_find')
        # parameters
        self.declare_parameter('raw_image', False)
        self.declare_parameter('image_topic', '/image_raw/compressed')
        self.declare_parameter('debug', True)
        img_topic = self.get_parameter('image_topic').value
        if self.get_parameter('raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
            self.sub1  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to raw image topic: {img_topic}')
        else:
            self.sub2 = self.create_subscription(CompressedImage, '/image_raw/compressed', self.listener_compressed, 10)
            self.sub2  # prevent unused variable warning
            self.get_logger().info(f'lane_detect subscribed to compressed image topic: {img_topic}')
        self.pub1 = self.create_publisher(Image, '/lane_img', 10)
        self.pub2 = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.debug = self.get_parameter('debug').value
        
    def listener_compressed(self, msg):
        # Convert ROS CompressedImage message to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print info of the image, only once not every time
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        
        
        gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        ros_image = self.bridge.cv2_to_imgmsg(gray, 'mono8')
        self.pub1.publish(ros_image)
        
def main(args=None):
    rclpy.init(args=args)
    paramfind = Paramfind()
    rclpy.spin(paramfind)
    paramfind.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()





