from robosuite.models.grippers import GRIPPER_MAPPING

from .suction_gripper import SuctionGripper


def register_suction_gripper():
    GRIPPER_MAPPING["SuctionGripper"] = SuctionGripper
    GRIPPER_MAPPING["suction"] = SuctionGripper


register_suction_gripper()
