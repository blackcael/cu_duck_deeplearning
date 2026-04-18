#!/usr/bin/env python3
"""
ROS1 skeleton node for deep-learning-based control.

Fill in:
- load_model()
- image_callback() preprocessing
- run_inference()
"""

import rospy
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import scripts.utils.model_utils as Model_Utils
import scripts.utils.fn_utils as FN_Utils

DUCKIEBOT = os.environ.get("VEHICLE_NAME")

WHEELS_CMD_TOPIC = f"/{DUCKIEBOT}/lane_controller_node/car_cmd"
IMAGE_TOPIC = f"/{DUCKIEBOT}/camera_node/image/compressed

X_VEL_PARAM_NAME = f"/{DUCKIEBOT}/dl_ctrl_x_vel"
X_VEL_DEF = 10

class DLControllerNode:
    def __init__(self):
        # Node params
        rospy.init_node("dl_controller_node", anonymous=True)

        self.bridge = CvBridge()
        self.latest_image = None
        self.model = Model_Utils.load_model(self.model_path)

        # Register Wheels Command Publisher
        self.cmd_pub = rospy.Publisher(
            WHEELS_CMD_TOPIC, 
            Twist, 
            queue_size=1
        )

        # Register Image Subscriber
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )

        # Params
        self.x_speed = rospy.set_param(X_VEL_PARAM_NAME, X_VEL_DEF)


    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_img
        except CvBridgeError as err:
            rospy.logwarn_throttle(2.0, "[%s] CvBridge error: %s", self.node_name, err)

    def run_inference(self, image_bgr):
        logits = self.model(image_bgr)
        angle = FN_Utils.logits_to_angle(logits)
        linear_x = self.x_speed
        angular_z = angle
        return linear_x, angular_z

    def spin(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            cmd = Twist()

            if self.latest_image is not None:
                linear_x, angular_z = self.run_inference(self.latest_image)
                cmd.linear.x = max(min(linear_x, self.max_linear_x), -self.max_linear_x)
                cmd.angular.z = max(min(angular_z, self.max_angular_z), -self.max_angular_z)

            self.cmd_pub.publish(cmd)
            rate.sleep()


def main():
    
    node = DLControllerNode()
    node.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
