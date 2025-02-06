import  cv2
import numpy as np
import warnings
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detect')
        self.start()
        self.setup_subscribers_and_publishers()
        self.initialize_pid_parameters()
        self.bridge = CvBridge()
        self.debug = False  # Set to True if yout want to enable debug mode
        self.there_is_right_lane = False
        self.there_is_left_lane = False

    def start(self):
        self.declare_parameter('is_raw_image', False)
        self.declare_parameter('raw_image_topic_name', '/camera/color/image_raw')
        self.declare_parameter('compressed_image_topic_name', '/image_raw/compressed')
        self.declare_parameter('debug', True)

    def setup_subscribers_and_publishers(self):
        img_topic = '/camera/color/image_raw' if self.get_parameter('is_raw_image').value else '/image_raw/compressed'
        if self.get_parameter('is_raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
            self.get_logger().info(f'lane_detect subscribed to raw image topic: {img_topic}')
        else:
            self.sub2 = self.create_subscription(CompressedImage, img_topic, self.compr_listener, 10)
            self.get_logger().info(f'lane_detect subscribed to compressed image topic: {img_topic}')
        self.lane_img_pub = self.create_publisher(Image, '/lane_img', 10)
        self.cannytest_lane_img_pub = self.create_publisher(Image, '/cannytest_lane_img', 10)
        self.brighttest_lane_img_pub = self.create_publisher(Image, '/brighttest_lane_img', 10)
        self.drive_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.offset_pub = self.create_publisher(Float32, '/lane_center_offset', 10)

    def initialize_pid_parameters(self): #  Sets up stuff for the actual driving part
        self.p_gain = 0.05 # Proportional gain
        self.i_gain = 0.005 # Integral gain
        self.d_gain = 0.02 # Derivative gain
        self.max_angle = 0.4  # Max steering angle in radians
        self.speed_mps = 0.5  # Speed in meters per second
        self.previous_t = self.get_clock().now() # Previous time
        self.previous_e = 0 # Previous error
        self.P, self.I, self.D = 0, 0, 0 # PID components

    def raw_listener(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info(f'First raw img arrived, shape: {cv_image.shape}', once=True)
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.lane_img_pub.publish(ros_image)
    
    def compr_listener(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.lane_img_pub.publish(ros_image)

    def detect_lanes(self, image):
        edges = self.apply_edge_detection(image)
        left_fitx, right_fitx, ploty, midpoint = self.sliding_window(edges)

        if not self.there_is_right_lane and not self.there_is_left_lane:
            self.get_logger().error("Failed to detect lanes.")
            return image

        center_deviation = self.calculate_center_deviation(left_fitx, right_fitx, midpoint)
        #self.publish_center_offset(center_deviation)
        self.callback(center_deviation)

        return self.visualize_lanes(image, left_fitx, right_fitx, ploty, center_deviation)

    def apply_edge_detection(self, image):
        # Crop the image to the bottom half
        height = image.shape[0]
        bottom_half = image[height//2:, :]

        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=10, beta=40) 
        gray = cv2.GaussianBlur(gray, (19, 13), 0)
        
        # Publish the image before Canny
        gray_ros_image = self.bridge.cv2_to_imgmsg(gray, 'mono8')
        self.brighttest_lane_img_pub.publish(gray_ros_image)
        
        edges = cv2.Canny(gray, 0, 50)
        
        # Publish the Canny edge-detected image
        edges_ros_image = self.bridge.cv2_to_imgmsg(edges, 'mono8')
        self.cannytest_lane_img_pub.publish(edges_ros_image)
        
        return edges

    def sliding_window(self, edges):
        window_height = 25
        num_windows = 8
        margin = 200
        minpix = 150

        self.there_is_right_lane = False
        self.there_is_left_lane = False

        histogram = np.sum(edges, axis=0)
        
        width = edges.shape[1]
        left_side = edges[:, :width // 2]
        histogram_l = np.sum(left_side, axis=1)
        
        right_side = edges[:, width // 2:]
        histogram_r = np.sum(right_side, axis=1)
        
        nonzero_l = np.nonzero(histogram_l)[0]
        first_nonzero_l = nonzero_l[0] if nonzero_l.size > 0 else None
        
        nonzero_r = np.nonzero(histogram_r)[0]
        first_nonzero_r = nonzero_r[0] if nonzero_r.size > 0 else None
        
        nonzero_l = np.nonzero(histogram_l)[0]
        last_nonzero_l = nonzero_l[-1] if nonzero_l.size > 0 else None
        
        nonzero_r = np.nonzero(histogram_r)[0]
        last_nonzero_r = nonzero_r[-1] if nonzero_r.size > 0 else None
        
        treshold = 40
        
        if first_nonzero_l is None or (last_nonzero_r is not None and last_nonzero_r - first_nonzero_l < treshold):
            self.there_is_right_lane = False
        else:
            self.there_is_right_lane = True
            
        if first_nonzero_r is None or (last_nonzero_l is not None and last_nonzero_l - first_nonzero_r < treshold):
            self.there_is_left_lane = False
        else:
            self.there_is_left_lane = True
        
        print(f"Last non-zero l-coordinate: {first_nonzero_l}")
        print(f"Last non-zero r-coordinate: {last_nonzero_r}")
        
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        '''
        # Determine if only one lane is visible
        left_peak = np.max(histogram[:midpoint])
        right_peak = np.max(histogram[midpoint:])
        visibility_threshold = 0.7 * np.max(histogram)

        self.there_is_left_lane = left_peak > visibility_threshold
        self.there_is_right_lane = right_peak > visibility_threshold
        
        self.there_is_left_lane = first_nonzero_l is not None and last_nonzero_l is not None
        self.there_is_right_lane = first_nonzero_r is not None and last_nonzero_r is not None
        '''
        if not self.there_is_left_lane and not self.there_is_right_lane:
            self.get_logger().error("No lanes detected.")
            return None, None, None, None

        nonzero = edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base if self.there_is_left_lane else None
        rightx_current = rightx_base if self.there_is_right_lane else None

        left_lane_inds = []
        right_lane_inds = []

        for window in range(num_windows):
            win_y_low = edges.shape[0] - (window + 1) * window_height
            win_y_high = edges.shape[0] - window * window_height

            if self.there_is_left_lane:
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))

            if self.there_is_right_lane:
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if self.there_is_left_lane else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if self.there_is_right_lane else np.array([])

        leftx = nonzerox[left_lane_inds] if self.there_is_left_lane else np.array([])
        lefty = nonzeroy[left_lane_inds] if self.there_is_left_lane else np.array([])
        rightx = nonzerox[right_lane_inds] if self.there_is_right_lane else np.array([])
        righty = nonzeroy[right_lane_inds] if self.there_is_right_lane else np.array([])

        # Debug statements
        if self.debug:
            print(f"Length of leftx: {len(leftx)}, Length of lefty: {len(lefty)}")
            print(f"Length of rightx: {len(rightx)}, Length of righty: {len(righty)}")

        # Check if there are enough points to fit a polynomial
        left_fit = None
        right_fit = None

        if len(leftx) > 0 and len(lefty) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.RankWarning)
                try:
                    left_fit = np.polyfit(lefty, leftx, 2)
                except np.RankWarning:
                    self.get_logger().warning("Left lane polyfit may be poorly conditioned")

        if len(rightx) > 0 and len(righty) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.RankWarning)
                try:
                    right_fit = np.polyfit(righty, rightx, 2)
                except np.RankWarning:
                    self.get_logger().warning("Right lane polyfit may be poorly conditioned")

        ploty = np.linspace(0, edges.shape[0] - 1, edges.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2] if left_fit is not None else None
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2] if right_fit is not None else None

        midpoint = edges.shape[1] // 2

        return left_fitx, right_fitx, ploty, midpoint

    def calculate_center_deviation(self, left_fitx, right_fitx, midpoint):
        if left_fitx is not None and right_fitx is not None:
            center_fitx_last = (left_fitx[-1] + right_fitx[-1]) / 2
            return (center_fitx_last - midpoint) / 255
        else:
            return 0.0
            
    def publish_center_offset(self, center_deviation):
        self.offset_pub.publish(Float32(data=center_deviation))

    def publish_twist_message(self, center_deviation):
        twist = Twist()
        twist.linear.x = self.speed_mps
        twist.angular.z = -self.p_gain * center_deviation
        self.drive_pub.publish(twist)

    def visualize_lanes(self, image, left_fitx, right_fitx, ploty, center_deviation):
        line_image = np.zeros_like(image)
        for i in range(len(ploty)):
            if self.there_is_left_lane and left_fitx is not None:
                cv2.circle(line_image, (int(left_fitx[i]), int(ploty[i] + image.shape[0]//2)), 2, (255, 0, 0), -1)
            if self.there_is_right_lane and right_fitx is not None:
                cv2.circle(line_image, (int(right_fitx[i]), int(ploty[i] + image.shape[0]//2)), 2, (0, 255, 0), -1)
            if self.there_is_left_lane and self.there_is_right_lane and left_fitx is not None and right_fitx is not None:
                cv2.circle(line_image, (int((left_fitx[i] + right_fitx[i]) / 2), int(ploty[i] + image.shape[0]//2)), 2, (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        direction = 'Straight'
        if center_deviation > 0.01:
            direction = 'Right'
        elif center_deviation < -0.01:
            direction = 'Left'
        cv2.putText(line_image, f'{direction} {abs(center_deviation):.2f}', (10, 30), font, 1, (60, 40, 200), 2, cv2.LINE_AA)

        if self.debug:
            return line_image
        else:
            return cv2.addWeighted(image, 0.8, line_image, 1, 1)

    def pid_control(self, error):
        
        t = self.get_clock().now()
        # Calculate the time difference (dt) in seconds
        dt = (t - self.previous_t).nanoseconds / 1e9

        # Calculate the proportional term
        self.P = self.p_gain * error

        # Calculate the integral term
        self.I += error * dt
        integral_term = self.i_gain * self.I

        # Calculate the derivative term
        derivative = (error - self.previous_e) / dt
        self.D = self.d_gain * derivative

        # Compute the PID output
        output = self.P + integral_term + self.D

        # Update previous error and time for the next iteration
        self.previous_e = error
        self.previous_t = t

        return output

    def callback(self, data):
        error = data
        
        steer_rad = self.pid_control(error)

        if steer_rad > self.max_angle or not self.there_is_right_lane:
            steer_rad = self.max_angle
            
        if steer_rad < -self.max_angle or not self.there_is_left_lane:
            steer_rad = -self.max_angle

        self.publish_center_offset(steer_rad)
        
        twist = Twist()
        twist.linear.x = self.speed_mps
        twist.angular.z = steer_rad

        self.drive_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    lane_detect = LaneDetector()
    rclpy.spin(lane_detect)
    lane_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()