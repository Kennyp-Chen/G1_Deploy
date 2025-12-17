"""
Base class for motion tracking policies
"""
from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import torch
import os


class MotionTrackingPolicy(FSMState):
    """Base class for policies that track reference motions"""

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput,
                 policy_name: FSMStateName, policy_name_str: str,
                 config_file: str, policy_dir: str):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = policy_name
        self.name_str = policy_name_str
        self.counter_step = 0

        config_path = os.path.join(policy_dir, "config", config_file)
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            model_name = config.get("pt_path", config.get("onnx_path"))
            self.model_path = os.path.join(policy_dir, model_name)
            self.kps_lab = np.array(config["kp_lab"], dtype=np.float32)
            self.kds_lab = np.array(config["kd_lab"], dtype=np.float32)
            self.default_angles_lab = np.array(
                config["default_angles_lab"], dtype=np.float32)
            self.mj2lab = np.array(config["mj2lab"], dtype=np.int32)
            self.action_scale_lab = np.array(
                config["action_scale_lab"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            
            # Auto-calculate motion length from CSV file and FPS
            motion_file = config.get("motion_file")
            fps = config.get("fps", 50)  # Default to 50 FPS if not specified
            
            if motion_file:
                csv_path = os.path.join(policy_dir, motion_file)
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as csv_f:
                        num_frames = sum(1 for _ in csv_f)
                    self.motion_length = num_frames / fps
                else:
                    # Fallback to config value if CSV not found
                    self.motion_length = config.get("motion_length", 140)
                    print(f"Warning: CSV file {motion_file} not found, using motion_length from config")
            else:
                # Fallback to config value if motion_file not specified
                self.motion_length = config.get("motion_length", 140)

            # Load policy model
            self.device = torch.device("cpu")
            self.policy = torch.jit.load(self.model_path).to(self.device)
            self.policy.eval()

            # Initialize buffers
            self.obs = torch.zeros(
                1, self.num_obs, dtype=torch.float32, device=self.device)
            self.time_step_tensor = torch.zeros(
                1, 1, dtype=torch.float32, device=self.device)
            self.action = np.zeros((1, self.num_actions), dtype=np.float32)
            self.ref_joint_pos = np.zeros(
                (1, self.num_actions), dtype=np.float32)
            self.ref_joint_vel = np.zeros(
                (1, self.num_actions), dtype=np.float32)
            self.ref_body_quat_w = np.zeros((1, 14, 4), dtype=np.float32)

            msg = f"{policy_name_str} policy initialized, "
            msg += f"motion length: {self.motion_length:.2f}s"
            print(msg)

    def quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])

    def matrix_from_quat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
        ])

    def yaw_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def euler_single_axis_to_quat(self, angle, axis, degrees=False):
        if degrees:
            angle = np.radians(angle)
        half_angle = angle * 0.5
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        if isinstance(axis, str):
            if axis.lower() == 'x':
                return np.array([cos_half, sin_half, 0.0, 0.0])
            elif axis.lower() == 'y':
                return np.array([cos_half, 0.0, sin_half, 0.0])
            elif axis.lower() == 'z':
                return np.array([cos_half, 0.0, 0.0, sin_half])
            else:
                raise ValueError("axis must be 'x', 'y', 'z'")
        else:
            axis = np.array(axis, dtype=np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError("axis vector cannot be zero")
            axis = axis / axis_norm
            w = cos_half
            x = sin_half * axis[0]
            y = sin_half * axis[1]
            z = sin_half * axis[2]
            return np.array([w, x, y, z])

    def enter(self):
        self.counter_step = 0
        self.action = np.zeros((1, self.num_actions), dtype=np.float32)
        with torch.no_grad():
            outputs = self.policy(self.obs, self.time_step_tensor)
            self.action = outputs[0].cpu().numpy()
            self.ref_joint_pos = outputs[1].cpu().numpy()
            self.ref_joint_vel = outputs[2].cpu().numpy()
            self.ref_body_quat_w = outputs[4].cpu().numpy()

    def run(self):
        robot_quat = self.state_cmd.base_quat
        qj = self.state_cmd.q[self.mj2lab]
        qj = (qj - self.default_angles_lab)

        base_troso_yaw = qj[2]
        base_troso_roll = qj[5]
        base_troso_pitch = qj[8]

        quat_yaw = self.euler_single_axis_to_quat(
            base_troso_yaw, 'z', degrees=False)
        quat_roll = self.euler_single_axis_to_quat(
            base_troso_roll, 'x', degrees=False)
        quat_pitch = self.euler_single_axis_to_quat(
            base_troso_pitch, 'y', degrees=False)
        temp1 = self.quat_mul(quat_roll, quat_pitch)
        temp2 = self.quat_mul(quat_yaw, temp1)
        robot_quat = self.quat_mul(robot_quat, temp2)
        ref_anchor_ori_w = self.ref_body_quat_w[:, 7].squeeze(0)

        if self.counter_step < 2:
            init_to_anchor = self.matrix_from_quat(
                self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(
                self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            self.counter_step += 1
            return

        motion_anchor_ori_b = (
            self.matrix_from_quat(robot_quat).T @
            self.init_to_world @
            self.matrix_from_quat(ref_anchor_ori_w)
        )

        ang_vel = self.state_cmd.ang_vel
        dqj = self.state_cmd.dq

        mimic_obs_buf = np.concatenate((
            self.ref_joint_pos.squeeze(0),
            self.ref_joint_vel.squeeze(0),
            motion_anchor_ori_b[:, :2].reshape(-1),
            ang_vel,
            qj,
            dqj[self.mj2lab],
            self.action.squeeze(0)
        ), axis=-1, dtype=np.float32)

        self.obs[0] = torch.from_numpy(mimic_obs_buf).float()
        self.time_step_tensor[0, 0] = float(self.counter_step)

        with torch.no_grad():
            outputs = self.policy(self.obs, self.time_step_tensor)
            self.action = outputs[0].cpu().numpy()
            self.ref_joint_pos = outputs[1].cpu().numpy()
            self.ref_joint_vel = outputs[2].cpu().numpy()
            self.ref_body_quat_w = outputs[4].cpu().numpy()

        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_lab = (
            self.action * self.action_scale_lab + self.default_angles_lab)
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab.squeeze(0)

        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab

        self.counter_step += 1
        motion_time = self.counter_step * 0.02
        motion_time = min(motion_time, self.motion_length)
        print(progress_bar(motion_time, self.motion_length),
              end="", flush=True)

    def exit(self):
        self.counter_step = 0
        print()

    def checkChange(self):
        motion_time = self.counter_step * 0.02

        if motion_time >= self.motion_length:
            return FSMStateName.SKILL_COOLDOWN

        # Only allow transitions to LOCO, PASSIVE, or POS_RESET
        # Ignore all skill commands (SKILL_1 to SKILL_9)
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        elif self.state_cmd.skill_cmd in [
            FSMCommand.SKILL_1, FSMCommand.SKILL_2, FSMCommand.SKILL_3,
            FSMCommand.SKILL_4, FSMCommand.SKILL_5, FSMCommand.SKILL_6,
            FSMCommand.SKILL_7, FSMCommand.SKILL_8, FSMCommand.SKILL_9
        ]:
            # Ignore skill switch commands, must go to LOCO first
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return self.name
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return self.name
