from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.kungfu.KungFu import KungFu
from policy.dance.Dance import Dance
from policy.skill_cooldown.SkillCooldown import SkillCooldown
from policy.skill_cast.SkillCast import SkillCast
from policy.kick.Kick import Kick
from policy.kungfu2.KungFu2 import KungFu2
from policy.fas_byd_7s.FasByd7s import FasByd7s
from policy.beyond_mimic_motions.dance101.Dance101 import Dance101
from policy.beyond_mimic_motions.dance102.Dance102 import Dance102
from policy.beyond_mimic_motions.fightAndSports101.FightAndSports101 import FightAndSports101
from policy.beyond_mimic_motions.run201.Run201 import Run201
from policy.beyond_mimic_motions.walk105.Walk105 import Walk105
from policy.beyond_mimic_motions.jump101.jump101 import jump101
from policy.beyond_mimic_motions.gangnam_style.GangnamStyle import GangnamStyle
from policy.beyond_mimic_motions.dance102_sar.Dance102Sar import Dance102Sar
from policy.beyond_mimic_motions.fallAndGetup101.FallAndGetup101 import FallAndGetup101
from policy.beyond_mimic_motions.dance204.Dance204 import Dance204
from FSM.FSMState import *
import time
from common.ctrlcomp import *
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.kungfu_policy = KungFu(state_cmd, policy_output)
        self.dance_policy = Dance(state_cmd, policy_output)
        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.skill_cast_policy = SkillCast(state_cmd, policy_output)
        self.kick_policy = Kick(state_cmd, policy_output)
        self.kungfu2_policy = KungFu2(state_cmd, policy_output)
        self.fas_byd_7s_policy = FasByd7s(state_cmd, policy_output)
        
        # Initialize new policies with error handling
        self.dance101_policy = None
        self.dance102_policy = None
        self.fightAndSports101_policy = None
        self.run201_policy = None
        self.walk105_policy = None
        
        try:
            print("Loading Dance101 policy...")
            self.dance101_policy = Dance101(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Dance101 policy: {e}")
            
        try:
            print("Loading Dance102 policy...")
            self.dance102_policy = Dance102(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Dance102 policy: {e}")
            
        try:
            print("Loading FightAndSports101 policy...")
            self.fightAndSports101_policy = FightAndSports101(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load FightAndSports101 policy: {e}")
            
        try:
            print("Loading Run201 policy...")
            self.run201_policy = Run201(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Run201 policy: {e}")
            
        try:
            print("Loading Walk105 policy...")
            self.walk105_policy = Walk105(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Walk105 policy: {e}")
        
        # Initialize additional new policies
        self.jump101_policy = None
        self.gangnam_style_policy = None
        self.dance102_sar_policy = None
        self.fallAndGetup101_policy = None
        self.dance204_policy = None
        
        try:
            print("Loading jump101 policy...")
            self.jump101_policy = jump101(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load jump101 policy: {e}")
        
        try:
            print("Loading GangnamStyle policy...")
            self.gangnam_style_policy = GangnamStyle(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load GangnamStyle policy: {e}")
        
        try:
            print("Loading Dance102Sar policy...")
            self.dance102_sar_policy = Dance102Sar(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Dance102Sar policy: {e}")
        
        try:
            print("Loading FallAndGetup101 policy...")
            self.fallAndGetup101_policy = FallAndGetup101(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load FallAndGetup101 policy: {e}")
        
        try:
            print("Loading Dance204 policy...")
            self.dance204_policy = Dance204(state_cmd, policy_output)
        except Exception as e:
            print(f"Warning: Failed to load Dance204 policy: {e}")
        
        print("initalized all policies!!!")
        
        self.cur_policy = self.passive_mode
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        start_time = time.time()
        if(self.FSMmode == FSMMode.NORMAL): 
            self.cur_policy.run()
            nextPolicyName = self.cur_policy.checkChange()
            
            if(nextPolicyName != self.cur_policy.name):
                # change policy
                self.FSMmode = FSMMode.CHANGE
                self.cur_policy.exit()
                self.get_next_policy(nextPolicyName)
                print("Switched to ", self.cur_policy.name_str)
        
        elif(self.FSMmode == FSMMode.CHANGE):
            self.cur_policy.enter()
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()
            
        # self.absoluteWait(self.cur_policy.control_horzion,self.start_time)
        end_time = time.time()
        # print("time cusume: ", end_time - start_time)

    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        if(policy_name == FSMStateName.PASSIVE):
            self.cur_policy = self.passive_mode
        elif((policy_name == FSMStateName.FIXEDPOSE)):
            self.cur_policy = self.fixed_pose_1
        elif((policy_name == FSMStateName.LOCOMODE)):
            self.cur_policy = self.loco_policy
        elif((policy_name == FSMStateName.SKILL_KungFu)):
            self.cur_policy = self.kungfu_policy
        elif((policy_name == FSMStateName.SKILL_Dance)):
            self.cur_policy = self.dance_policy
        elif((policy_name == FSMStateName.SKILL_COOLDOWN)):
            self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_CAST)):
            self.cur_policy = self.skill_cast_policy
        elif((policy_name == FSMStateName.SKILL_KICK)):
            self.cur_policy = self.kick_policy
        elif((policy_name == FSMStateName.SKILL_KungFu2)):
            self.cur_policy = self.kungfu2_policy
        elif((policy_name == FSMStateName.SKILL_FAS_BYD_7S)):
            self.cur_policy = self.fas_byd_7s_policy
        elif((policy_name == FSMStateName.SKILL_Dance101)):
            if self.dance101_policy is not None:
                self.cur_policy = self.dance101_policy
            else:
                print("Dance101 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_Dance102)):
            if self.dance102_policy is not None:
                self.cur_policy = self.dance102_policy
            else:
                print("Dance102 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_FightAndSports101)):
            if self.fightAndSports101_policy is not None:
                self.cur_policy = self.fightAndSports101_policy
            else:
                print("FightAndSports101 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_Run201)):
            if self.run201_policy is not None:
                self.cur_policy = self.run201_policy
            else:
                print("Run201 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_Walk105)):
            if self.walk105_policy is not None:
                self.cur_policy = self.walk105_policy
            else:
                print("Walk105 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_jump101)):
            if self.jump101_policy is not None:
                self.cur_policy = self.jump101_policy
            else:
                print("jump101 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_GANGNAM_STYLE)):
            if self.gangnam_style_policy is not None:
                self.cur_policy = self.gangnam_style_policy
            else:
                print("GangnamStyle policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_DANCE102_SAR)):
            if self.dance102_sar_policy is not None:
                self.cur_policy = self.dance102_sar_policy
            else:
                print("Dance102Sar policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_FALLANDGETUP101)):
            if self.fallAndGetup101_policy is not None:
                self.cur_policy = self.fallAndGetup101_policy
            else:
                print("FallAndGetup101 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_DANCE204)):
            if self.dance204_policy is not None:
                self.cur_policy = self.dance204_policy
            else:
                print("Dance204 policy not available, switching to cooldown")
                self.cur_policy = self.skill_cooldown_policy
        else:
            pass
            
        
        