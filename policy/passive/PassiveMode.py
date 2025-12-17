from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMStateName, FSMCommand
import os

class PassiveMode(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.PASSIVE
        self.name_str = "passive_mode"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "Passive.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.kds = np.array(config["kds"], dtype=np.float32)
    
    def enter(self):
        self.policy_output.kps = np.zeros(self.state_cmd.num_joints)
        self.policy_output.kds = self.kds.copy()
    
    def run(self):
        kps = np.zeros(self.state_cmd.num_joints)
        kds = self.kds.copy()
        actions = np.zeros(self.state_cmd.num_joints)
        
        self.policy_output.actions = actions.copy()
        self.policy_output.kps = kps.copy()
        self.policy_output.kds = kds.copy()
    
    def exit(self):
        self.policy_output.kps = np.zeros(self.state_cmd.num_joints)
        self.policy_output.kds = self.kds.copy()
        
    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        elif(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_1):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_CAST
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_2):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_CAST
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_3):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_KICK
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_4):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_CAST
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_5):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_Dance101
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_6):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_Dance102
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_7):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_FightAndSports101
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_8):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_Run201
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_9):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_Walk105
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        