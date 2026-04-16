LR = 0.01
AXEL_LENGTH_MM = 50

import torch



def velocity_pair_to_angle(v_l, v_r):
    return (v_r - v_l) / AXEL_LENGTH_MM

def angle_to_velocity_pair(angle, avg_velocity):
    v_l = avg_velocity - (angle * AXEL_LENGTH_MM /2)
    v_r = avg_velocity + (angle * AXEL_LENGTH_MM /2)
    return (v_l, v_r)

def scale_v_pair_to_45(v_pair, max_mag_w, axel_len_mm):
    """
    takes a value between 0 and 45. 23 is the middle value, so in reality we are scaling from -22 to 0 to 22
    """
    w = (v_pair[0] - v_pair[1]) / axel_len_mm
    unit_w = w / max_mag_w
    return(round(unit_w * 22))

def logits_to_angle(logits):
    return torch.argmax(logits, dim=1) - 22