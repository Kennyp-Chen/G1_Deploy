from common.path_config import PROJECT_ROOT

import numpy as np
from enum import Enum, unique

@unique
class FSMStateName(Enum):
    INVALID = -1
    PASSIVE = 1
    FIXEDPOSE = 2
    SKILL_COOLDOWN = 3
    LOCOMODE = 4
    SKILL_CAST = 5
    SKILL_KungFu = 6
    SKILL_Dance = 7
    SKILL_KICK = 8
    SKILL_KungFu2 = 9
    SKILL_FAS_BYD_7S = 10
    SKILL_Dance101 = 11
    SKILL_Dance102 = 12
    SKILL_FightAndSports101 = 13
    SKILL_Run201 = 14
    SKILL_Walk105 = 15
    SKILL_jump101 = 16
    SKILL_GANGNAM_STYLE = 17
    SKILL_DANCE102_SAR = 18
    SKILL_FALLANDGETUP101 = 19
    SKILL_DANCE204 = 20


@unique
class FSMCommand(Enum):
    INVALID = -1
    POS_RESET = 1
    LOCO = 2
    PASSIVE = 4
    SKILL_1 = 5
    SKILL_2 = 6
    SKILL_3 = 7
    SKILL_4 = 8
    SKILL_5 = 9
    SKILL_6 = 10
    SKILL_7 = 11
    SKILL_8 = 12
    SKILL_9 = 13
    SKILL_10 = 14
    SKILL_11 = 15
    SKILL_12 = 16
    SKILL_13 = 17
    SKILL_14 = 18

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def progress_bar(current, total, length=50):
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "-" * (length - filled)
    return f"\r|{bar}| {percent:.1%} [{current:.3f}s/{total:.3f}s]"

def scale_values(values, target_ranges):
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled)


