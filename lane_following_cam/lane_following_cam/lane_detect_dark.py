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
        self.debug = False  # Set to True if you want to enable debug mode
        self.there_is_right_lane = False
        self.there_is_left_lane = False
        self.size = (0, 0)
        self.frame_count = 0  # Initialize frame counter
        self.left_fitx = np.zeros(3)  # Initialize as array
        self.right_fitx = np.zeros(3)
        self.lane_width = 300

    def start(self):
        self.declare_parameter('is_raw_image', True)
        self.declare_parameter('raw_image_topic_name', '/image_raw')
        self.declare_parameter('compressed_image_topic_name', '/image_raw/compressed')
        self.declare_parameter('debug', True)

    def setup_subscribers_and_publishers(self):
        img_topic = '/image_raw' if self.get_parameter('is_raw_image').value else '/image_raw/compressed'
        if self.get_parameter('is_raw_image').value:
            self.sub1 = self.create_subscription(Image, img_topic, self.raw_listener, 10)
            self.get_logger().info(f'lane_detect subscribed to raw image topic: {img_topic}')
        else:
            self.sub2 = self.create_subscription(CompressedImage, img_topic, self.compr_listener, 10)
            self.get_logger().info(f'lane_detect subscribed to compressed image topic: {img_topic}')
        self.lane_img_pub = self.create_publisher(Image, '/lane_img', 10)
        self.amask_cannytest_lane_img_pub = self.create_publisher(Image, '/amask_cannytest_lane_img', 10)
        self.brighttest_lane_img_pub = self.create_publisher(Image, '/brighttest_lane_img', 10)
        self.drive_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.offset_pub = self.create_publisher(Float32, '/lane_center_offset', 10)
        
    def initialize_pid_parameters(self):
        self.p_gain = 0.00125
        self.speed_mps = 0.1
        self.max_angle = 0.4

    def raw_listener(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info(f'First raw img arrived, shape: {cv_image.shape}', once=True)
        self.size = cv_image.shape
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.lane_img_pub.publish(ros_image)
    
    def compr_listener(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.get_logger().info(f'First compressed img arrived, shape: {cv_image.shape}', once=True)
        self.size = cv_image.shape
        lane_image = self.detect_lanes(cv_image)
        ros_image = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.lane_img_pub.publish(ros_image)

    def detect_lanes(self, image):
        print(self.left_fitx, self.right_fitx)
        edges = self.apply_edge_detection(image)
        
        self.lane_check(edges)
        
        if not self.there_is_left_lane and not self.there_is_right_lane:
            self.get_logger().error("No lanes detected.")
            return np.zeros_like(image)
        
        self.left_fitx, self.right_fitx = self.sliding_window(edges)
        left_fitx, right_fitx = self.left_fitx, self.right_fitx
        
        midpoint = image.shape[1] // 2

        center_deviation = self.calculate_center_deviation(left_fitx, right_fitx, midpoint)
        self.publish_center_offset(center_deviation)
        self.publish_twist_message(center_deviation)

        return self.visualize_lanes(image, left_fitx, right_fitx, center_deviation)

    def basic_canny_edge_detection(self, image):
        # Crop the image to the bottom half
        height = image.shape[0]
        bottom_half = image[int(height/1.8):, :]

        #Greyscale
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=10, beta=20) 
        gray = cv2.GaussianBlur(gray, (15, 19), 0)
        
        # Publish the image before Canny
        gray_ros_image = self.bridge.cv2_to_imgmsg(gray, 'mono8')
        self.brighttest_lane_img_pub.publish(gray_ros_image)
        
        final_edges = cv2.Canny(gray, 0, 20)
        return final_edges
        
    def edge_refinement(self, edges):
        # Keep only the highest y values per x value
        highest_y_edges = np.zeros_like(edges)
        for x in range(edges.shape[1]):
            y_indices = np.where(edges[:, x] > 0)[0]
            if len(y_indices) > 0:
                highest_y = np.max(y_indices)
                highest_y_edges[highest_y, x] = 255
        
        # Post-process edges to close gaps and reduce noise
        kernel = np.ones((5, 7), np.uint8)
        highest_y_edges = cv2.morphologyEx(highest_y_edges, cv2.MORPH_CLOSE, kernel)
        
        # Remove small detected edges
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(highest_y_edges, connectivity=8)
        min_area = 3
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                highest_y_edges[labels == i] = 0

        return highest_y_edges

    def apply_edge_detection(self, image):
        
        edges = self.basic_canny_edge_detection(image)
        
        final_edges = self.edge_refinement(edges)
        
        # Publish the Canny edge-detected image
        edges_ros_image = self.bridge.cv2_to_imgmsg(final_edges, 'mono8')
        self.amask_cannytest_lane_img_pub.publish(edges_ros_image)

        return final_edges

    def lane_check(self, edges):
        
        self.there_is_left_lane = False
        self.there_is_right_lane = False
        
        width = edges.shape[1]
        left_side = edges[:, :width // 2]
        histogram_l = np.sum(left_side, axis=1)
        
        right_side = edges[:, width // 2:]
        histogram_r = np.sum(right_side, axis=1)
        
        nonzero_l = np.nonzero(histogram_l)[0]
        first_nonzero_l = nonzero_l[0] if nonzero_l.size > 0 else None
        last_nonzero_l = nonzero_l[-1] if nonzero_l.size > 0 else None
        
        nonzero_r = np.nonzero(histogram_r)[0]
        first_nonzero_r = nonzero_r[0] if nonzero_r.size > 0 else None
        last_nonzero_r = nonzero_r[-1] if nonzero_r.size > 0 else None
        
        lane_existing_treshold = 10
        left_exist_condition = first_nonzero_l is not None and last_nonzero_r - first_nonzero_l < lane_existing_treshold
        right_exist_condition = first_nonzero_r is not None and last_nonzero_l - first_nonzero_r < lane_existing_treshold
        
        if first_nonzero_l is None or left_exist_condition:
            self.there_is_right_lane = False
        else:
            self.there_is_right_lane = True
            
        if first_nonzero_r is None or right_exist_condition:
            self.there_is_left_lane = False
        else:
            self.there_is_left_lane = True

    def fit_polynomial(self, leftx, lefty, rightx, righty, edges):
        left_fit = None
        right_fit = None

        if len(leftx) > 0 and len(lefty) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.RankWarning)
                try:
                    left_fit = np.polyfit(lefty, leftx, 2)
                except (np.RankWarning, np.linalg.LinAlgError):
                    self.get_logger().warning("Left lane polyfit may be poorly conditioned")

        if len(rightx) > 0 and len(righty) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('error', np.RankWarning)
                try:
                    right_fit = np.polyfit(righty, rightx, 2)
                except (np.RankWarning, np.linalg.LinAlgError):
                    self.get_logger().warning("Right lane polyfit may be poorly conditioned")

        ploty = np.linspace(0, edges.shape[0] - 1, edges.shape[0])

        if self.there_is_left_lane and left_fit is not None:
            
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        else:
            left_fitx = self.create_imaginary_lane(right_fit, self.lane_width, 'left')

        if self.there_is_right_lane and right_fit is not None:
            
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        else:
            right_fitx = self.create_imaginary_lane(left_fit, self.lane_width, 'right')

        return left_fitx, right_fitx
        
    def sliding_window(self, edges):
        window_height = edges.shape[0] // 10
        num_windows = edges.shape[0] // window_height
        margin = edges.shape[1] // 4
        minpix = window_height
        
        histogram = np.sum(edges, axis=0)
        
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base if self.there_is_left_lane else None
        rightx_current = rightx_base if self.there_is_right_lane else None

        left_lane_inds = []
        right_lane_inds = []

        max_leftx = np.full(edges.shape[0], np.inf)
        max_rightx = np.full(edges.shape[0], -np.inf)

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
                
                if len(good_left_inds) < minpix:
                    margin += 25  # Expand search in the next window
                else:
                    margin = edges.shape[1] // 4  # Reset to default margin

            if self.there_is_right_lane:
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
                if len(good_right_inds) < minpix:
                    margin += 25  # Expand search in the next window
                else:
                    margin = edges.shape[1] // 4  # Reset to default margin

        left_lane_inds = np.concatenate(left_lane_inds) if self.there_is_left_lane else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if self.there_is_right_lane else np.array([])

        leftx = nonzerox[left_lane_inds] if self.there_is_left_lane else np.array([])
        lefty = nonzeroy[left_lane_inds] if self.there_is_left_lane else np.array([])
        rightx = nonzerox[right_lane_inds] if self.there_is_right_lane else np.array([])
        righty = nonzeroy[right_lane_inds] if self.there_is_right_lane else np.array([])

        # Filter out the infinite values
        valid_lefty = np.where(max_leftx != np.inf)[0]
        valid_righty = np.where(max_rightx != np.inf)[0]
        max_leftx = max_leftx[valid_lefty]
        max_rightx = max_rightx[valid_righty]

        return self.fit_polynomial(leftx, lefty, rightx, righty, edges)

    def sanity_check(self, left_fitx, right_fitx):
        # Check curvature similarity
        y_eval = self.size[0]  # Bottom of the image
        
        left_curverad = ((1 + (2 * left_fitx[0] * y_eval + left_fitx[1]) ** 2) ** 1.5 / np.abs(2 * left_fitx[0]))
        right_curverad = ((1 + (2 * right_fitx[0] * y_eval + right_fitx[1]) ** 2) ** 1.5 / np.abs(2 * right_fitx[0]))
        curvature_diff = np.abs(left_curverad - right_curverad)

        # Check lane width consistency
        lane_width = np.abs(left_fitx[2] - right_fitx[2])
        return curvature_diff < 1000 and 300 < lane_width < 500  # Adjust thresholds

    def create_imaginary_lane(self, fit, lane_width, direction):
        if fit is None:
            return None

        ploty = np.linspace(0, self.size[0] - 1, self.size[0])
        
        if direction == 'left':
            imaginary_fitx = fit[0] * ploty**2 + fit[1] * ploty + (fit[2] - lane_width)
        else:
            imaginary_fitx = fit[0] * ploty**2 + fit[1] * ploty + (fit[2] + lane_width)

        return imaginary_fitx

    def targeted_search(self, edges, left_fitx_prev, right_fitx_prev):
        nonzero = edges.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        margin = 50  # Adjust margin as needed

        # Handle invalid previous fits (initialize to 3 coefficients for 2nd-order)
        if left_fitx_prev is None or len(left_fitx_prev) < 3:
            left_fitx_prev = np.zeros(3)
        if right_fitx_prev is None or len(right_fitx_prev) < 3:
            right_fitx_prev = np.zeros(3)

        left_lane_inds = []
        right_lane_inds = []

        if self.there_is_left_lane:
            # 2nd-order polynomial evaluation
            left_lane_inds = (
                (nonzerox > (left_fitx_prev[0] * nonzeroy**2 + left_fitx_prev[1] * nonzeroy + left_fitx_prev[2] - margin)) &
                (nonzerox < (left_fitx_prev[0] * nonzeroy**2 + left_fitx_prev[1] * nonzeroy + left_fitx_prev[2] + margin))
            )

        if self.there_is_right_lane:
            # 2nd-order polynomial evaluation
            right_lane_inds = (
                (nonzerox > (right_fitx_prev[0] * nonzeroy**2 + right_fitx_prev[1] * nonzeroy + right_fitx_prev[2] - margin)) &
                (nonzerox < (right_fitx_prev[0] * nonzeroy**2 + right_fitx_prev[1] * nonzeroy + right_fitx_prev[2] + margin))
            )

        leftx = nonzerox[left_lane_inds] if self.there_is_left_lane else np.array([])
        lefty = nonzeroy[left_lane_inds] if self.there_is_left_lane else np.array([])
        rightx = nonzerox[right_lane_inds] if self.there_is_right_lane else np.array([])
        righty = nonzeroy[right_lane_inds] if self.there_is_right_lane else np.array([])

        return self.fit_polynomial(leftx, lefty, rightx, righty, edges)

    def process_frame(self, edges):
        if self.frame_count == 5 or self.right_fitx is None or self.left_fitx is None:
            # Reset with sliding window every 5 frames
            self.left_fitx, self.right_fitx = self.sliding_window(edges)
            self.frame_count = 0
        else:
            self.left_fitx, self.right_fitx = self.targeted_search(edges, self.left_fitx, self.right_fitx)
            self.frame_count += 1  # Increment frame counter

        '''# Validate and smooth
        if self.sanity_check(left_fitx, right_fitx):
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx'''

        return self.left_fitx, self.right_fitx
    
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
        twist.angular.z = self.p_gain * center_deviation
        
        if twist.angular.z < -self.max_angle or not self.there_is_right_lane:
            twist.angular.z = -self.max_angle
            
        if twist.angular.z > self.max_angle or not self.there_is_left_lane:
            twist.angular.z = self.max_angle
        
        self.drive_pub.publish(twist)

    def visualize_lanes(self, image, left_fitx, right_fitx, center_deviation):
        line_image = np.zeros_like(image)
        height = image.shape[0]
        offset = int(height / 1.8)
        for i in range(height - offset):
            if self.there_is_left_lane and left_fitx is not None:
                cv2.circle(line_image, (int(left_fitx[i]), int(i + offset)), 2, (255, 0, 0), -1)
            if self.there_is_right_lane and right_fitx is not None:
                cv2.circle(line_image, (int(right_fitx[i]), int(i + offset)), 2, (0, 255, 0), -1)
            #if self.there_is_left_lane and self.there_is_right_lane and left_fitx is not None and right_fitx is not None:
                #cv2.circle(line_image, (int((left_fitx[i] + right_fitx[i]) / 2), int(i + offset)), 2, (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        direction = 'Straight'
        if center_deviation > 0.01:
            direction = 'Right'
        elif center_deviation < -0.01:
            direction = 'Left'
        cv2.putText(line_image, f'{direction} {abs(center_deviation):.2f}', (10, 30), font, 1, (60, 40, 200), 2, cv2.LINE_AA)

        return cv2.addWeighted(image, 0.8, line_image, 1, 1)

def main(args=None):
    rclpy.init(args=args)
    lane_detect = LaneDetector()
    rclpy.spin(lane_detect)
    lane_detect.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()