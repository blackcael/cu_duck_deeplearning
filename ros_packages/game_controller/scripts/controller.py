#!/usr/bin/env python3
import pygame
import rospy
from sensor_msgs.msg import Joy

DEFAULT_AXIS_MAP = [0, 1, 2, 3]
DEFAULT_BUTTON_COUNT_FALLBACK = 16

def _deadzone(value, dz):
    return 0.0 if abs(value) < dz else value

def _read_axis(js, axis_index, deadzone):
    if axis_index < 0 or axis_index >= js.get_numaxes():
        return 0.0
    return _deadzone(js.get_axis(axis_index), deadzone)

def _pick_joystick(preferred_name_substring):
    count = pygame.joystick.get_count()
    if count == 0:
        return None

    lowered_pref = preferred_name_substring.lower().strip()
    if lowered_pref:
        for idx in range(count):
            candidate = pygame.joystick.Joystick(idx)
            candidate.init()
            name = candidate.get_name() or ""
            if lowered_pref in name.lower():
                return candidate
            candidate.quit()

    js = pygame.joystick.Joystick(0)
    js.init()
    return js

def main():
    rospy.init_node("gamepad_to_joy")
    pub = rospy.Publisher("joy", Joy, queue_size=1)
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        rospy.logerr("No controller detected.")
        return

    rate_hz = rospy.get_param("~rate_hz", 20)
    preferred_name = rospy.get_param("~preferred_name", "8BitDo")
    axis_map = rospy.get_param("~axis_map", DEFAULT_AXIS_MAP)
    invert_y = rospy.get_param("~invert_y", True)
    deadzone = rospy.get_param("~deadzone", 0.08)

    js = _pick_joystick(preferred_name)
    if js is None:
        rospy.logerr("No controller detected.")
        return

    rospy.loginfo("Using joystick: '%s' (axes=%d, buttons=%d)",
                  js.get_name(), js.get_numaxes(), js.get_numbuttons())
    rospy.loginfo("Controller params: preferred_name='%s', axis_map=%s, invert_y=%s, deadzone=%.3f",
                  preferred_name, axis_map, str(invert_y), deadzone)

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        pygame.event.pump()

        msg = Joy()
        msg.header.stamp = rospy.Time.now()

        # Axes layout defaults to [LX, LY, RX, RY].
        axes = [_read_axis(js, int(i), deadzone) for i in axis_map]
        if len(axes) < 4:
            axes.extend([0.0] * (4 - len(axes)))
        if invert_y:
            axes[1] *= -1.0
            axes[3] *= -1.0
        msg.axes = axes

        button_count = js.get_numbuttons() if js.get_numbuttons() > 0 else DEFAULT_BUTTON_COUNT_FALLBACK
        msg.buttons = [js.get_button(i) for i in range(button_count)]

        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
