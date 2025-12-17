import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from pynput import keyboard
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.joystick import JoyStick, JoystickButton, JoystickAxis

sys.stderr = open(os.devnull, 'w')


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def print_state(state_cmd, elastic_band):
    rope_status = "ON" if elastic_band.enable else "OFF"
    print(f"X: {state_cmd.vel_cmd[0]:.2f}, Y: {state_cmd.vel_cmd[1]:.2f}, "
          f"Yaw: {state_cmd.vel_cmd[2]:.2f} | "
          f"Rope: {rope_status}, H: {elastic_band.point[2]:.2f}m, "
          f"L: {elastic_band.length:.2f}m", end='\r')


class ElasticBand:
    """弹性绳索控制器 - 使用弹性力而不是约束"""
    def __init__(self):
        self.stiffness = 200  # 刚度
        self.damping = 100    # 阻尼
        self.point = np.array([0, 0, 3.0])  # 锚点位置
        self.length = 0.0     # 松弛长度（0表示完全拉紧）
        self.enable = True    # 是否启用
        self.height_increment = 0.1
        print("Elastic rope system initialized")
    
    def advance(self, x, dx):
        """
        计算弹性绳索施加的力
        Args:
            x: 当前位置
            dx: 当前速度
        Returns:
            force: 施加的力向量
        """
        if not self.enable:
            return np.zeros(3)
        
        # 计算从当前位置到锚点的向量
        delta_x = self.point - x
        distance = np.linalg.norm(delta_x)
        
        if distance < 1e-6:  # 避免除零
            return np.zeros(3)
        
        direction = delta_x / distance
        
        # 计算沿绳索方向的速度
        v = np.dot(dx, direction)
        
        # 弹性力：F = k * (distance - length) - d * v
        force_magnitude = (self.stiffness * (distance - self.length) - 
                          self.damping * v)
        force = force_magnitude * direction
        
        return force
    
    def set_height(self, height):
        """设置绳索锚点高度"""
        self.point[2] = np.clip(height, 0.5, 5.0)
    
    def increase_height(self):
        """升高绳索"""
        self.point[2] += self.height_increment
        self.point[2] = min(self.point[2], 5.0)
        print(f"\nRope height increased to {self.point[2]:.2f}m")
    
    def decrease_height(self):
        """降低绳索"""
        self.point[2] -= self.height_increment
        self.point[2] = max(self.point[2], 0.5)
        print(f"\nRope height decreased to {self.point[2]:.2f}m")
    
    def increase_length(self):
        """增加松弛长度（放松绳索）"""
        self.length += 0.1
        self.length = min(self.length, 2.0)
        print(f"\nRope length increased to {self.length:.2f}m")
    
    def decrease_length(self):
        """减少松弛长度（拉紧绳索）"""
        self.length -= 0.1
        self.length = max(self.length, 0.0)
        print(f"\nRope length decreased to {self.length:.2f}m")
    
    def toggle(self):
        """切换绳索启用/禁用"""
        self.enable = not self.enable
        status = "enabled" if self.enable else "disabled"
        print(f"\nRope {status}")


class KeyboardState:
    def __init__(self):
        self.pressed_keys = set()
        self.just_pressed = set()
    
    def update(self):
        self.just_pressed.clear()
    
    def is_just_pressed(self, char):
        return char in self.just_pressed
    
    def press(self, char):
        if char not in self.pressed_keys:
            self.pressed_keys.add(char)
            self.just_pressed.add(char)
    
    def release(self, char):
        self.pressed_keys.discard(char)


def handle_keyboard_input(state_cmd, kb_state, elastic_band):
    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                kb_state.press(key.char)
                # Skill commands
                if key.char == 'r':
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                elif key.char == '0':
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                elif key.char == '1':
                    state_cmd.skill_cmd = FSMCommand.LOCO
                elif key.char == '2':
                    state_cmd.skill_cmd = FSMCommand.SKILL_1
                elif key.char == '3':
                    state_cmd.skill_cmd = FSMCommand.SKILL_2
                elif key.char == '4':
                    state_cmd.skill_cmd = FSMCommand.SKILL_3
                elif key.char == '5':
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                elif key.char == '6':
                    state_cmd.skill_cmd = FSMCommand.SKILL_5
                elif key.char == '7':
                    state_cmd.skill_cmd = FSMCommand.SKILL_6
                elif key.char == '8':
                    state_cmd.skill_cmd = FSMCommand.SKILL_7
                elif key.char == '9':
                    state_cmd.skill_cmd = FSMCommand.SKILL_8
                elif key.char == 'p':
                    state_cmd.skill_cmd = FSMCommand.SKILL_9
                elif key.char == 'o':
                    state_cmd.skill_cmd = FSMCommand.SKILL_10
                elif key.char == 'i':
                    state_cmd.skill_cmd = FSMCommand.SKILL_11
                elif key.char == 'u':
                    state_cmd.skill_cmd = FSMCommand.SKILL_12
                elif key.char == 'y':
                    state_cmd.skill_cmd = FSMCommand.SKILL_13
                elif key.char == 't':
                    state_cmd.skill_cmd = FSMCommand.SKILL_14
                # Rope controls
                elif key.char == '-':
                    elastic_band.decrease_length()  # 拉紧绳索
                elif key.char == '=':
                    elastic_band.increase_length()  # 放松绳索
                elif key.char == '[':
                    elastic_band.decrease_height()  # 降低高度
                elif key.char == ']':
                    elastic_band.increase_height()  # 升高高度
                elif key.char == '\\':
                    elastic_band.toggle()  # 开关绳索
        except AttributeError:
            pass
    
    def on_release(key):
        try:
            if hasattr(key, 'char') and key.char:
                kb_state.release(key.char)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def update_velocity_from_keyboard(state_cmd, kb_state):
    """Update velocity commands"""
    vel_increment = 0.1
    
    if kb_state.is_just_pressed('w'):
        state_cmd.vel_cmd[0] += vel_increment
        print(f"\nW: Forward {state_cmd.vel_cmd[0]:.2f}")
    if kb_state.is_just_pressed('s'):
        state_cmd.vel_cmd[0] -= vel_increment
        print(f"\nS: Backward {state_cmd.vel_cmd[0]:.2f}")
    if kb_state.is_just_pressed('a'):
        state_cmd.vel_cmd[1] += vel_increment
        print(f"\nA: Left {state_cmd.vel_cmd[1]:.2f}")
    if kb_state.is_just_pressed('d'):
        state_cmd.vel_cmd[1] -= vel_increment
        print(f"\nD: Right {state_cmd.vel_cmd[1]:.2f}")
    if kb_state.is_just_pressed('q'):
        state_cmd.vel_cmd[2] += vel_increment
        print(f"\nQ: Rotate Left {state_cmd.vel_cmd[2]:.2f}")
    if kb_state.is_just_pressed('e'):
        state_cmd.vel_cmd[2] -= vel_increment
        print(f"\nE: Rotate Right {state_cmd.vel_cmd[2]:.2f}")
    if kb_state.is_just_pressed(' '):
        state_cmd.vel_cmd = np.zeros(3)
        print("\nSpace: Reset velocities")
    
    kb_state.update()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用普通场景文件（弹性绳索不需要特殊场景）
    xml_path = os.path.join(PROJECT_ROOT, "g1_description", "scene.xml")
    
    # 加载配置
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
    
    # 加载模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    # 初始化控制器
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    # 初始化弹性绳索
    elastic_band = ElasticBand()
    
    # 尝试初始化手柄
    joystick_available = False
    try:
        joystick = JoyStick()
        joystick_available = True
    except RuntimeError:
        print("No joystick connected. Using keyboard control only.")
    
    # 初始化键盘
    kb_state = KeyboardState()
    kb_listener = handle_keyboard_input(state_cmd, kb_state, elastic_band)
    
    print("\n" + "="*70)
    print("MuJoCo Deployment - Control Mapping")
    print("="*70)
    print("\n【基本控制】")
    print("  键盘          手柄")
    print("  1  = LOCO    | B  = LOCO")
    print("  R  = PASSIVE | X  = PASSIVE")
    print("  0  = 重置     | A  = 重置")
    print("  Ctrl+C = 退出 | Y  = 退出")
    print("\n【运动控制】(LOCO模式)")
    print("  键盘: W/S=前后, A/D=左右, Q/E=转向, Space=停止")
    print("  手柄: 左摇杆=移动, 右摇杆=转向")
    print("\n【技能触发】")
    print("  键盘: 2-4=基础, 5-9/P=BeyondMimic1-5, O/I/U/Y/T=BeyondMimic6-10")
    print("  手柄: R1+X/Y/B=基础, L1+X/Y/B/A=BM1-4, L2+X/Y/B/A=BM5-8, R2+X/Y/B=BM9-11")
    print("\n【绳索控制】")
    print("  [/] = 降低/升高, -/= = 拉紧/放松, \\ = 开关")
    print("="*70 + "\n")
    
    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and Running:
            try:
                # 手柄控制
                if joystick_available:
                    joystick.update()
                    
                    # 检测L2/R2是否被按下
                    l2_pressed = joystick.is_trigger_pressed(JoystickAxis.L2)
                    r2_pressed = joystick.is_trigger_pressed(JoystickAxis.R2)
                    r1_pressed = joystick.is_button_pressed(JoystickButton.R1)
                    l1_pressed = joystick.is_button_pressed(JoystickButton.L1)
                    
                    # 基本控制（单独按键，不与其他组合键同时按下）
                    if joystick.is_button_released(JoystickButton.Y) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                        Running = False
                        print("\nY button pressed, exiting...")
                    if joystick.is_button_released(JoystickButton.X) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                        state_cmd.skill_cmd = FSMCommand.PASSIVE
                    if joystick.is_button_released(JoystickButton.A) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                        state_cmd.skill_cmd = FSMCommand.POS_RESET
                    if joystick.is_button_released(JoystickButton.B) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                        state_cmd.skill_cmd = FSMCommand.LOCO
                    
                    # R1组 - 基础技能
                    if r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                        if joystick.is_button_released(JoystickButton.X):
                            state_cmd.skill_cmd = FSMCommand.SKILL_1
                        elif joystick.is_button_released(JoystickButton.Y):
                            state_cmd.skill_cmd = FSMCommand.SKILL_2
                        elif joystick.is_button_released(JoystickButton.B):
                            state_cmd.skill_cmd = FSMCommand.SKILL_3
                    
                    # L1组 - BeyondMimic 1-4
                    elif l1_pressed and not r1_pressed and not l2_pressed and not r2_pressed:
                        if joystick.is_button_released(JoystickButton.X):
                            state_cmd.skill_cmd = FSMCommand.SKILL_4
                        elif joystick.is_button_released(JoystickButton.Y):
                            state_cmd.skill_cmd = FSMCommand.SKILL_5
                        elif joystick.is_button_released(JoystickButton.B):
                            state_cmd.skill_cmd = FSMCommand.SKILL_6
                        elif joystick.is_button_released(JoystickButton.A):
                            state_cmd.skill_cmd = FSMCommand.SKILL_7
                    
                    # L2组 - BeyondMimic 5-8
                    elif l2_pressed and not r1_pressed and not l1_pressed and not r2_pressed:
                        if joystick.is_button_released(JoystickButton.X):
                            state_cmd.skill_cmd = FSMCommand.SKILL_8
                        elif joystick.is_button_released(JoystickButton.Y):
                            state_cmd.skill_cmd = FSMCommand.SKILL_9
                        elif joystick.is_button_released(JoystickButton.B):
                            state_cmd.skill_cmd = FSMCommand.SKILL_10
                        elif joystick.is_button_released(JoystickButton.A):
                            state_cmd.skill_cmd = FSMCommand.SKILL_11
                    
                    # R2组 - BeyondMimic 9-11
                    elif r2_pressed and not r1_pressed and not l1_pressed and not l2_pressed:
                        if joystick.is_button_released(JoystickButton.X):
                            state_cmd.skill_cmd = FSMCommand.SKILL_12
                        elif joystick.is_button_released(JoystickButton.Y):
                            state_cmd.skill_cmd = FSMCommand.SKILL_13
                        elif joystick.is_button_released(JoystickButton.B):
                            state_cmd.skill_cmd = FSMCommand.SKILL_14
                    
                    state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                    state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                    state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                # 键盘控制
                is_loco_mode = (state_cmd.skill_cmd == FSMCommand.LOCO)
                if is_loco_mode:
                    update_velocity_from_keyboard(state_cmd, kb_state)
                else:
                    kb_state.update()
                
                step_start = time.time()
                
                # 计算并施加弹性绳索力
                if elastic_band.enable:
                    # 使用基座的位置和速度（前3个元素）
                    base_pos = d.qpos[:3]
                    base_vel = d.qvel[:3]
                    rope_force = elastic_band.advance(base_pos, base_vel)
                    
                    # 将力施加到躯干
                    torso_id = mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
                    d.xfrc_applied[torso_id, :3] = rope_force
                
                # PD 控制
                tau = pd_control(policy_output_action, d.qpos[7:], kps,
                                np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1
                
                if sim_counter % control_decimation == 0:
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
                    
                    # 更新状态显示
                    print_state(state_cmd, elastic_band)
            
            except ValueError as e:
                print(str(e))
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
