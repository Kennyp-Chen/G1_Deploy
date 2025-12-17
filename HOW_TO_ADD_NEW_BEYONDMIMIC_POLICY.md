# 如何添加新的 BeyondMimic 策略

## 快速指南

假设你要添加一个名为 `NewMotion` 的新策略，按照以下步骤操作：

---

## 步骤 1：准备文件

### 1.1 创建文件夹结构

```bash
mkdir -p policy/beyond_mimic_motions/new_motion/config
mkdir -p policy/beyond_mimic_motions/new_motion/model
```

### 1.2 放置模型文件

将训练好的模型文件放入 `model/` 文件夹：
```
policy/beyond_mimic_motions/new_motion/model/your_model.pt
```

---

## 步骤 2：创建配置文件

创建 `policy/beyond_mimic_motions/new_motion/config/NewMotion.yaml`：

```yaml
# 模型文件路径
pt_path: "model/your_model.pt"

# 动作数据文件（用于自动计算时长）
motion_file: "50fps_your_motion.csv"  # CSV动作文件名
fps: 50  # 动作数据的帧率（默认50，根据实际情况设置）

# 基本参数
num_actions: 29
num_obs: 154
# 注意：motion_length 会根

# PD 控制器增益
kp_lab: [200.0, 200.0, 200.0, 300.0, 40.0, 200.0, 200.0, 200.0, 300.0, 40.0, 300.0, 100.0, 100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 60.0, 60.0, 60.0]

kd_lab: [5.0, 5.0, 5.0, 6.0, 2.0, 5.0, 5.0, 5.0, 6.0, 2.0, 6.0, 2.0, 2.0, 2.0, 2.0, 6.0, 2.0, 2.0, 2.0, 6.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0]

# 默认关节角度
default_angles_lab: [0.0, 0.0, 0.0, 0.4, -0.8, 0.0, 0.0, 0.0, 0.4, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 动作缩放因子
action_scale_lab: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

# 坐标系映射
mj2lab: [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28]
```

---

## 步骤 3：创建策略类

创建 `policy/beyond_mimic_motions/new_motion/NewMotion.py`：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from FSM.FSMState import FSMStateName
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.motion_tracking_base import MotionTrackingPolicy
import os


class NewMotion(MotionTrackingPolicy):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(
            state_cmd=state_cmd,
            policy_output=policy_output,
            policy_name=FSMStateName.SKILL_NewMotion,
            policy_name_str="skill_new_motion",
            config_file="NewMotion.yaml",
            policy_dir=current_dir
        )
```

**注意**：
- 类名：`NewMotion`（首字母大写）
- `policy_name`：使用 `FSMStateName.SKILL_NewMotion`
- `policy_name_str`：小写加下划线 `"skill_new_motion"`
- `config_file`：配置文件名 `"NewMotion.yaml"`

---

## 步骤 4：修改代码文件

### 4.1 修改 `common/utils.py`

在 `FSMStateName` 枚举中添加新状态：

```python
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
    SKILL_NewMotion = 16  # ← 添加这一行，使用下一个可用数字
```

### 4.2 修改 `FSM/FSM.py`

#### 4.2.1 添加导入（文件开头）

```python
from policy.beyond_mimic_motions.new_motion.NewMotion import NewMotion
```

#### 4.2.2 在 `__init__` 方法中初始化策略

找到其他策略的初始化代码，添加类似的代码：

```python
def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
    # ... 现有代码 ...
    
    # 在这里添加新策略初始化
    try:
        print("Loading NewMotion policy...")
        self.new_motion_policy = NewMotion(state_cmd, policy_output)
    except Exception as e:
        print(f"Warning: Failed to load NewMotion policy: {e}")
```

#### 4.2.3 在 `get_next_policy` 方法中添加切换逻辑

找到其他策略的切换代码，添加类似的代码：

```python
def get_next_policy(self, policy_name:FSMStateName):
    # ... 现有代码 ...
    
    elif((policy_name == FSMStateName.SKILL_NewMotion)):
        if self.new_motion_policy is not None:
            self.cur_policy = self.new_motion_policy
        else:
            print("NewMotion policy not available, switching to cooldown")
            self.cur_policy = self.skill_cooldown_policy
    
    # ... 其他代码 ...
```

---

## 步骤 5：添加键盘映射（可选）

如果需要通过键盘触发新策略：

### 5.1 添加 FSMCommand（如果需要新按键）

编辑 `common/utils.py`：

```python
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
    SKILL_10 = 14  # ← 添加新命令
```

### 5.2 添加键盘映射

编辑 `deploy_mujoco/deploy_mujoco_kb.py`，在 `on_press` 函数中添加：

```python
elif key.char == 'o':  # 选择一个未使用的按键
    state_cmd.skill_cmd = FSMCommand.SKILL_10
```

### 5.3 在 LocoMode 中添加切换

编辑 `policy/loco_mode/LocoMode.py` 的 `checkChange` 方法：

```python
def checkChange(self):
    # ... 现有代码 ...
    
    elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_10):
        return FSMStateName.SKILL_NewMotion
    
    # ... 其他代码 ...
```

---

## 步骤 6：测试

```bash
python deploy_mujoco/deploy_mujoco_kb.py
```

测试流程：
1. 按 `1` 进入 LOCO 模式
2. 按对应的按键（如 `o`）触发新策略
3. 观察机器人执行动作
4. 动作完成后应自动回到 LOCO 模式

---

## 完整的文件修改清单

### 必须修改的文件：

1. ✅ **创建策略文件**
   - `policy/beyond_mimic_motions/new_motion/NewMotion.py`
   - `policy/beyond_mimic_motions/new_motion/config/NewMotion.yaml`
   - `policy/beyond_mimic_motions/new_motion/model/your_model.pt`

2. ✅ **`common/utils.py`**
   - 添加 `FSMStateName.SKILL_NewMotion`

3. ✅ **`FSM/FSM.py`**
   - 添加导入
   - 在 `__init__` 中初始化
   - 在 `get_next_policy` 中添加切换逻辑

### 可选修改的文件（如果需要键盘控制）：

4. ⭕ **`common/utils.py`**
   - 添加 `FSMCommand.SKILL_10`（如果需要新按键）

5. ⭕ **`deploy_mujoco/deploy_mujoco_kb.py`**
   - 添加键盘映射

6. ⭕ **`policy/loco_mode/LocoMode.py`**
   - 在 `checkChange` 中添加切换逻辑

---

## 代码模板总结

### 策略类模板

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from FSM.FSMState import FSMStateName
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.motion_tracking_base import MotionTrackingPolicy
import os


class YourPolicyName(MotionTrackingPolicy):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(
            state_cmd=state_cmd,
            policy_output=policy_output,
            policy_name=FSMStateName.SKILL_YourPolicyName,
            policy_name_str="skill_your_policy_name",
            config_file="YourPolicyName.yaml",
            policy_dir=current_dir
        )
```

### FSM 初始化模板

```python
# 在 FSM.__init__ 中
try:
    print("Loading YourPolicyName policy...")
    self.your_policy_name_policy = YourPolicyName(state_cmd, policy_output)
except Exception as e:
    print(f"Warning: Failed to load YourPolicyName policy: {e}")
```

### FSM 切换模板

```python
# 在 FSM.get_next_policy 中
elif((policy_name == FSMStateName.SKILL_YourPolicyName)):
    if self.your_policy_name_policy is not None:
        self.cur_policy = self.your_policy_name_policy
    else:
        print("YourPolicyName policy not available, switching to cooldown")
        self.cur_policy = self.skill_cooldown_policy
```

---

## 常见问题

### Q1: 模型加载失败
**A**: 检查模型文件路径和格式（必须是 TorchScript `.pt` 或 ONNX `.onnx`）

### Q2: 无法切换到新策略
**A**: 确认以下内容：
- `FSMStateName` 已添加
- `FSM.py` 中的导入、初始化、切换逻辑都已添加
- 键盘映射正确（如果使用）

### Q3: 机器人动作异常
**A**: 调整配置文件中的参数：
- `kp_lab` / `kd_lab` - PD 控制器增益
- `action_scale_lab` - 动作缩放因子
- `motion_length` - 动作时长

### Q4: 导入错误
**A**: 确保策略文件中的路径设置正确：
```python
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
```
注意是 **三个** `parent`，因为在 `beyond_mimic_motions` 子文件夹中。

---

## 当前已有的策略

| 按键 | FSMCommand | FSMStateName | 文件夹 |
|------|-----------|--------------|--------|
| 5 | SKILL_4 | SKILL_FAS_BYD_7S | `fas_byd_7s/` |
| 6 | SKILL_5 | SKILL_Dance101 | `beyond_mimic_motions/dance101/` |
| 7 | SKILL_6 | SKILL_Dance102 | `beyond_mimic_motions/dance102/` |
| 8 | SKILL_7 | SKILL_FightAndSports101 | `beyond_mimic_motions/fightAndSports101/` |
| 9 | SKILL_8 | SKILL_Run201 | `beyond_mimic_motions/run201/` |
| P | SKILL_9 | SKILL_Walk105 | `beyond_mimic_motions/walk105/` |

下一个可用的：
- FSMStateName: `SKILL_NewMotion = 16`
- FSMCommand: `SKILL_10 = 14`
- 按键: `o`, `i`, `u`, `y`, `t` 等

---

## 相关文档

- `BEYONDMIMIC_POLICY_GUIDE.md` - 详细指南
- `REORGANIZATION_SUMMARY.md` - 重组总结
- `policy/motion_tracking_base.py` - 基类实现
