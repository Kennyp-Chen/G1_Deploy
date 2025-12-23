import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import RemoteController, KeyMap
from config import Config
import threading
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: 'keyboard' library not available. Keyboard control disabled.")
    print("Install with: pip install keyboard")


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


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()
        self.num_joints = config.num_joints
        self.control_dt = config.control_dt
        
        # 初始化键盘状态
        self.kb_state = KeyboardState()
        self.setup_keyboard_listener()
        
        
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        
        # inital connection
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.wait_for_low_state()
        
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        
        self.policy_output_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kps = np.zeros(self.num_joints, dtype=np.float32)
        self.kds = np.zeros(self.num_joints, dtype=np.float32)
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0, 0, -1], dtype=np.float32)
        
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.running = True
        self.counter_over_time = 0
        self.loop_counter = 0  # 用于控制打印频率
    
    def setup_keyboard_listener(self):
        """设置键盘监听器"""
        if not KEYBOARD_AVAILABLE:
            print("Keyboard control disabled - library not available")
            return None
        
        try:
            def keyboard_handler(event):
                try:
                    if event.event_type == 'down':
                        self.kb_state.press(event.name)
                        # Skill commands
                        if event.name == 'r':
                            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
                        elif event.name == '0':
                            self.state_cmd.skill_cmd = FSMCommand.POS_RESET
                        elif event.name == '1':
                            self.state_cmd.skill_cmd = FSMCommand.LOCO
                        elif event.name == '2':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_1
                        elif event.name == '3':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_2
                        elif event.name == '4':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_3
                        elif event.name == '5':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_4
                        elif event.name == '6':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_5
                        elif event.name == '7':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_6
                        elif event.name == '8':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_7
                        elif event.name == '9':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_8
                        elif event.name == 'p':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_9
                        elif event.name == 'o':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_10
                        elif event.name == 'i':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_11
                        elif event.name == 'u':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_12
                        elif event.name == 'y':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_13
                        elif event.name == 't':
                            self.state_cmd.skill_cmd = FSMCommand.SKILL_14
                    elif event.event_type == 'up':
                        self.kb_state.release(event.name)
                except Exception as e:
                    print(f"Keyboard handler error: {e}")
            
            keyboard.hook(keyboard_handler)
            print("Keyboard control enabled")
            return True
        except Exception as e:
            print(f"Failed to setup keyboard listener: {e}")
            print("Keyboard control disabled - requires root privileges")
            return None
    
    def update_velocity_from_keyboard(self):
        """从键盘更新速度命令（仅在LOCO模式下）"""
        if not KEYBOARD_AVAILABLE:
            return
            
        vel_increment = 0.1
        
        if self.kb_state.is_just_pressed('w'):
            self.state_cmd.vel_cmd[0] += vel_increment
        if self.kb_state.is_just_pressed('s'):
            self.state_cmd.vel_cmd[0] -= vel_increment
        if self.kb_state.is_just_pressed('a'):
            self.state_cmd.vel_cmd[1] += vel_increment
        if self.kb_state.is_just_pressed('d'):
            self.state_cmd.vel_cmd[1] -= vel_increment
        if self.kb_state.is_just_pressed('q'):
            self.state_cmd.vel_cmd[2] += vel_increment
        if self.kb_state.is_just_pressed('e'):
            self.state_cmd.vel_cmd[2] -= vel_increment
        if self.kb_state.is_just_pressed('space'):
            self.state_cmd.vel_cmd = np.zeros(3)
        
        self.kb_state.update()
        
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        
    def run(self):
        try:
            # if(self.counter_over_time >= config.error_over_time):
            #     raise ValueError("counter_over_time >= error_over_time")
            
            loop_start_time = time.time()
            
            # 检测组合键状态
            r1_pressed = self.remote_controller.is_button_pressed(KeyMap.R1)
            l1_pressed = self.remote_controller.is_button_pressed(KeyMap.L1)
            l2_pressed = self.remote_controller.is_button_pressed(KeyMap.L2)
            r2_pressed = self.remote_controller.is_button_pressed(KeyMap.R2)
            
            # 手柄基本控制（单独按键，不与组合键同时按下）
            if self.remote_controller.is_button_pressed(KeyMap.X) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.A) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.B) and not r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            
            # R1组 - 基础技能
            if r1_pressed and not l1_pressed and not l2_pressed and not r2_pressed:
                if self.remote_controller.is_button_pressed(KeyMap.X):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_1
                elif self.remote_controller.is_button_pressed(KeyMap.Y):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_2
                elif self.remote_controller.is_button_pressed(KeyMap.B):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_3
            
            # L1组 - BeyondMimic 1-4
            elif l1_pressed and not r1_pressed and not l2_pressed and not r2_pressed:
                if self.remote_controller.is_button_pressed(KeyMap.X):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_4
                elif self.remote_controller.is_button_pressed(KeyMap.Y):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_5
                elif self.remote_controller.is_button_pressed(KeyMap.B):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_6
                elif self.remote_controller.is_button_pressed(KeyMap.A):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_7
            
            # L2组 - BeyondMimic 5-8
            elif l2_pressed and not r1_pressed and not l1_pressed and not r2_pressed:
                if self.remote_controller.is_button_pressed(KeyMap.X):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_8
                elif self.remote_controller.is_button_pressed(KeyMap.Y):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_9
                elif self.remote_controller.is_button_pressed(KeyMap.B):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_10
                elif self.remote_controller.is_button_pressed(KeyMap.A):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_11
            
            # R2组 - BeyondMimic 9-11
            elif r2_pressed and not r1_pressed and not l1_pressed and not l2_pressed:
                if self.remote_controller.is_button_pressed(KeyMap.X):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_12
                elif self.remote_controller.is_button_pressed(KeyMap.Y):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_13
                elif self.remote_controller.is_button_pressed(KeyMap.B):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_14
            
            # 手柄速度控制
            self.state_cmd.vel_cmd[0] = self.remote_controller.ly
            self.state_cmd.vel_cmd[1] = self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] = self.remote_controller.rx * -1
            
            # 键盘速度控制（仅在LOCO模式下）
            is_loco_mode = (self.FSM_controller.cur_policy.name == FSMStateName.LOCOMODE)
            if is_loco_mode:
                self.update_velocity_from_keyboard()
            else:
                self.kb_state.update()

            for i in range(self.num_joints):
                self.qj[i] = self.low_state.motor_state[i].q
                self.dqj[i] = self.low_state.motor_state[i].dq

            # imu_state quaternion: w, x, y, z
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
            
            gravity_orientation = get_gravity_orientation_real(quat)
            
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = ang_vel.copy()
            self.state_cmd.base_quat = quat
            
            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()
            
            # Build low cmd
            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
                
            # send the command
            # create_damping_cmd(controller.low_cmd) # only for debug
            self.send_cmd(self.low_cmd)
            
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            
            self.loop_counter += 1
            
            if delta_time < self.control_dt:
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                # 每50次循环（1秒）只打印一次警告
                if self.loop_counter % 50 == 0:
                    print(f"Warning: control loop over time. Avg: {delta_time*1000:.1f}ms (target: {self.control_dt*1000:.1f}ms)")
                self.counter_over_time += 1
            pass
        except ValueError as e:
            print(str(e))
            pass
        
        pass
        
        
if __name__ == "__main__":
    config = Config()
    # Initialize DDS communication
    ChannelFactoryInitialize(0, config.net)
    
    controller = Controller(config)
    
    while True:
        try:
            controller.run()
            # Press Y key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.Y) and \
               not controller.remote_controller.is_button_pressed(KeyMap.R1) and \
               not controller.remote_controller.is_button_pressed(KeyMap.L1) and \
               not controller.remote_controller.is_button_pressed(KeyMap.L2) and \
               not controller.remote_controller.is_button_pressed(KeyMap.R2):
                print("\nY button pressed, exiting...")
                break
        except KeyboardInterrupt:
            break
    
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    