import os
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel


class SuctionGripper(GripperModel):
    def __init__(self, idn=0):
        xml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "assets",
                "grippers",
                "suction_gripper.xml",
            )
        )
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        action = np.array(action, dtype=np.float32).reshape(-1)
        assert len(action) == self.dof
        self.current_action = np.clip(action, -1.0, 1.0)
        # 第 7 维 action_suction 只作为吸盘“开/关”的逻辑指令使用：
        # - bddl_base_domain.py 里的 _update_suction(action) 会直接读取原始 action[6]；
        # - 这里如果把 action_suction 继续传给 MuJoCo position actuator，
        #   suction_joint 会在 [-0.001, 0.001] 范围内移动，导致同样前 6 维动作下接触力不同。
        # 因此这里保留 self.current_action 方便调试，但给真实滑动关节的控制固定为 0，
        # 让吸盘 cup 始终停在中位，只由 weld/latch 逻辑决定是否吸附物体。
        return np.zeros(self.dof, dtype=np.float32)

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        return np.array([0.0], dtype=np.float32)
