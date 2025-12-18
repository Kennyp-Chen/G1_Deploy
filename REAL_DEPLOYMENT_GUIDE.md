# G1 机器人真机部署指南

> 从 MuJoCo 仿真到真实机器人的完整部署流程

---

## 📋 目录

1. [部署方式选择](#部署方式选择)
2. [部署前准备](#部署前准备)
3. [环境配置](#环境配置)
4. [安全检查](#安全检查)
5. [测试流程](#测试流程)
6. [故障排查](#故障排查)
7. [创建虚拟环境](#使用-venv创建虚拟环境)
---

## 🚀 部署方式选择

### 两种部署方式对比

#### 方式1：外部电脑控制
```
外部笔记本 ──网络──> 机器人
   运行代码              接收命令
```

**优点：**
- ✅ 开发调试方便
- ✅ 可以随时修改代码
- ✅ 计算资源充足
- ✅ 实时查看日志

**缺点：**
- ❌ 网络延迟（5-20ms）
- ❌ 可能丢包
- ❌ 依赖网络稳定性
- ❌ 需要携带笔记本

**适用场景：** 开发和调试阶段

#### 方式2：机器人主机运行（推荐）✅
```
机器人主机
   运行代码 + 控制硬件（本地通信）
```

**优点：**
- ✅ 零网络延迟（本地IPC，< 1ms）
- ✅ 更稳定可靠
- ✅ 独立运行，无需外部设备
- ✅ 更适合实际应用和演示

**缺点：**
- ⚠️ 主机计算资源可能有限
- ⚠️ 调试稍微不便（需要SSH）

**适用场景：** 生产环境和演示

### 部署到机器人主机的步骤

#### 1. 连接到机器人
```bash
# SSH 登录到机器人主机
ssh unitree@<机器人IP>
# 默认密码通常是 123 或 unitree
```

#### 2. 传输项目文件

**方法1：使用 scp**
```bash
# 从你的电脑执行
scp -r /path/to/RoboMimic_Deploy unitree@<机器人IP>:~/
```

**方法2：使用 rsync（推荐，支持增量传输）**
```bash
# 从你的电脑执行
rsync -avz --progress /path/to/RoboMimic_Deploy unitree@<机器人IP>:~/

# 后续更新只传输修改的文件
rsync -avz --progress --delete /path/to/RoboMimic_Deploy unitree@<机器人IP>:~/
```

**方法3：使用 git（如果机器人有网络）**
```bash
# 在机器人上执行
ssh unitree@<机器人IP>
git clone https://github.com/Kennyp-Chen/G1_Deploy.git
cd G1_Deploy
```

#### 3. 安装依赖（在机器人上）
```bash
# SSH 登录到机器人
ssh unitree@<机器人IP>

# 检查 Python 环境
python3 --version

# 安装依赖
pip3 install numpy pynput
# unitree_sdk2py 通常已预装在机器人上

# 如果使用手柄
pip3 install pygame  # 或其他手柄库
```

#### 4. 修改配置（关键步骤）

**编辑配置文件：**
```bash
# 在机器人主机上
vim ~/RoboMimic_Deploy/deploy_real/config.py
# 或
nano ~/RoboMimic_Deploy/deploy_real/config.py
```

**关键修改：**
```python
class Config:
    def __init__(self):
        # 本地通信，使用 lo（loopback）或实际接口
        self.net = "lo"  # 或 "eth0"，取决于 DDS 配置
        
        # 其他配置保持不变
        self.control_dt = 0.02
        self.num_joints = 29
        self.lowcmd_topic = "rt/lowcmd"
        self.lowstate_topic = "rt/lowstate"
```

**网络配置对比：**
```python
# 外部控制（通过网络）
self.net = "eth0"  # 以太网接口

# 主机运行（本地通信）
self.net = "lo"    # 本地回环接口
# 或保持 "eth0"，取决于 DDS 配置
```

#### 5. 运行测试
```bash
# 在机器人主机上
cd ~/RoboMimic_Deploy
python3 deploy_real/deploy_real.py
```

#### 6. 后台运行（可选）
```bash
# 使用 screen 或 tmux 保持会话
screen -S robot
python3 deploy_real/deploy_real.py
# 按 Ctrl+A, D 分离会话

# 重新连接
screen -r robot

# 或使用 nohup
nohup python3 deploy_real/deploy_real.py > robot.log 2>&1 &
```

### 混合方案（最佳实践）

**开发流程：**
```bash
# 1. 在外部电脑开发和初步测试
python deploy_real/deploy_real.py

# 2. 测试通过后，同步到机器人
rsync -avz --progress RoboMimic_Deploy unitree@<IP>:~/

# 3. 在机器人上运行
ssh unitree@<IP>
cd ~/RoboMimic_Deploy
python3 deploy_real/deploy_real.py

# 4. 如需调试，通过 SSH 查看日志
ssh unitree@<IP> "tail -f ~/robot.log"
```

### 自动启动配置（可选）

**创建 systemd 服务：**
```bash
# 在机器人上创建服务文件
sudo vim /etc/systemd/system/robot-control.service
```

**服务配置：**
```ini
[Unit]
Description=Robot Control Service
After=network.target

[Service]
Type=simple
User=unitree
WorkingDirectory=/home/unitree/RoboMimic_Deploy
ExecStart=/usr/bin/python3 deploy_real/deploy_real.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**启用服务：**
```bash
sudo systemctl daemon-reload
sudo systemctl enable robot-control.service
sudo systemctl start robot-control.service

# 查看状态
sudo systemctl status robot-control.service

# 查看日志
journalctl -u robot-control.service -f
```

### 性能对比

| 指标 | 外部控制 | 主机运行 |
|------|----------|----------|
| 通信延迟 | 5-20ms | < 1ms |
| 稳定性 | 依赖网络 | 非常稳定 |
| 丢包率 | 0.1-1% | 0% |
| 开发便利性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 生产可靠性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 注意事项

1. **计算资源**
   - 确认机器人主机 CPU/内存足够运行策略推理
   - 监控 CPU 使用率，避免过载

2. **文件大小**
   - 项目包含大量 `.pt` 和 `.onnx` 文件（~178MB）
   - 首次传输需要时间，建议使用 rsync

3. **权限**
   - 确保在机器人上有执行权限
   - 可能需要 sudo 权限访问某些硬件接口

4. **调试**
   - 使用 SSH 远程调试
   - 配置日志输出到文件
   - 使用 screen/tmux 保持会话

5. **更新代码**
   - 使用 rsync 增量同步
   - 或使用 git pull 更新

---

## 🔧 部署前准备

### 1. 硬件检查

**机器人状态：**
- [ ] 电池电量 > 80%
- [ ] 所有关节活动正常，无异响
- [ ] IMU 传感器工作正常
- [ ] 急停按钮功能正常
- [ ] 机器人外观无损坏

**测试环境：**
- [ ] 平坦、防滑的地面（至少 4m × 4m）
- [ ] 周围无障碍物（半径 3m 内）
- [ ] 光线充足
- [ ] 有安全人员在场

**控制设备：**
- [ ] 笔记本电脑电量充足（外部控制）
- [ ] 手柄已连接并测试（如使用）
- [ ] 网络连接稳定

### 2. 软件检查

```bash
# 检查依赖
python -c "import unitree_sdk2py; print('SDK OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import pynput; print('Pynput OK')"

# 检查手柄（可选）
python -c "from common.joystick import JoyStick; j = JoyStick(); print('Joystick OK')"
```

### 3. 配置文件检查

**检查网络配置：** `deploy_real/config.py`

```python
class Config:
    def __init__(self):
        self.net = "eth0"  # 确认网络接口名称
        self.control_dt = 0.02  # 50Hz 控制频率
        self.num_joints = 29  # G1 关节数
        # ...
```

**验证网络接口：**
```bash
# 查看可用网络接口
ifconfig
# 或
ip addr show

# 测试与机器人的连接（外部控制时）
ping <机器人IP>
```

---

## 🌐 环境配置

### 1. 网络连接（外部控制）

**有线连接（推荐）：**
```bash
# 设置静态IP（如需要）
sudo ifconfig eth0 192.168.123.100 netmask 255.255.255.0

# 验证连接
ping 192.168.123.10  # 机器人默认IP
```

**无线连接：**
- 确保在同一局域网
- 延迟 < 10ms
- 信号强度 > -60dBm

### 2. DDS 配置

```bash
# 设置 DDS 域 ID（如需要）
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/dds_config.xml

# 或在代码中配置
# ChannelFactoryInitialize(0, "eth0")
```

### 3. 权限设置

```bash
# 如果需要 sudo 权限
sudo usermod -a -G dialout $USER
# 重新登录使权限生效
```

---

## 🛡️ 安全检查

### 关键安全措施

**1. 紧急停止方式（按优先级）：**

```
优先级 1: 物理急停按钮（机器人背部）
优先级 2: 键盘 R 键 → PASSIVE 模式
优先级 3: 手柄 X 按钮 → PASSIVE 模式
优先级 4: Ctrl+C → 退出程序
```

**2. 安全距离：**
- 操作人员距离机器人 > 2m
- 观察人员距离机器人 > 3m
- 技能执行时保持安全距离

**3. 监控指标：**
```python
# 程序会自动监控以下指标
- 控制循环超时次数（counter_over_time）
- 关节位置和速度
- IMU 数据有效性
```

**4. 测试前确认：**
- [ ] 所有人员已知晓紧急停止方式
- [ ] 清空测试区域
- [ ] 准备好记录设备（可选）
- [ ] 确认机器人初始姿态正常

---

## 🚀 测试流程

### 阶段 1：连接测试（5分钟）

```bash
# 1. 启动程序
cd /path/to/RoboMimic_Deploy
python deploy_real/deploy_real.py

# 预期输出：
# Successfully connected to the robot.
# Enter zero torque state.
# Waiting for the start signal...
```

**验证点：**
- [ ] 程序成功连接到机器人
- [ ] 无错误信息
- [ ] 控制循环频率正常（无 "control loop over time"）

### 阶段 2：PASSIVE 模式测试（5分钟）

**目的：** 验证阻尼控制和紧急停止

```
操作步骤：
1. 按 R 键进入 PASSIVE 模式
2. 轻轻推动机器人手臂，感受阻尼
3. 测试手柄 X 按钮（如使用手柄）
4. 验证机器人保持平衡
```

**验证点：**
- [ ] 关节有适当阻尼（不会自由摆动）
- [ ] 机器人能保持站立
- [ ] 紧急停止响应及时

### 阶段 3：位置重置测试（10分钟）

**目的：** 验证从任意姿态回到初始位置

```
操作步骤：
1. 在 PASSIVE 模式下，手动调整机器人姿态
2. 按 0 键（键盘）或 A 按钮（手柄）
3. 观察机器人平稳回到初始姿态
4. 重复 3-5 次，测试不同初始姿态
```

**验证点：**
- [ ] 运动平滑，无突变
- [ ] 最终姿态一致
- [ ] 无关节抖动
- [ ] 过程中可随时按 R 键停止

**⚠️ 注意：** 如果出现剧烈抖动或异常运动，立即按急停！

### 阶段 4：LOCO 模式测试（20分钟）

**目的：** 验证基本运动控制

#### 4.1 静态测试（5分钟）

```
操作步骤：
1. 按 1 键（键盘）或 B 按钮（手柄）进入 LOCO 模式
2. 不给速度命令，观察机器人保持站立
3. 持续 30 秒，检查稳定性
```

**验证点：**
- [ ] 机器人稳定站立
- [ ] 无漂移
- [ ] 无异常抖动

#### 4.2 小幅度运动（10分钟）

```
键盘控制：
1. 按 W 键 1 次（前进速度 +0.1）
2. 观察机器人缓慢前进
3. 按 Space 键停止
4. 重复测试：S（后退）、A（左移）、D（右移）
5. 测试转向：Q（左转）、E（右转）

手柄控制：
1. 轻推左摇杆（前后左右）
2. 轻推右摇杆（转向）
3. 回中停止
```

**验证点：**
- [ ] 响应及时（< 0.1s）
- [ ] 运动方向正确
- [ ] 速度可控
- [ ] 停止平稳

#### 4.3 组合运动（5分钟）

```
操作步骤：
1. 前进 + 左转（W + Q 或左摇杆↑ + 右摇杆←）
2. 后退 + 右转（S + E）
3. 侧移 + 转向（A/D + Q/E）
```

**验证点：**
- [ ] 多轴运动协调
- [ ] 无失衡
- [ ] 轨迹符合预期

**⚠️ 速度限制建议：**
- 前后速度：-0.5 ~ 0.5 m/s
- 侧向速度：-0.3 ~ 0.3 m/s
- 转向速度：-0.5 ~ 0.5 rad/s

### 阶段 5：技能测试（60分钟+）

**测试顺序：从短到长，从简单到复杂**

#### 5.1 短时长技能（15分钟）

```
1. FAS_BYD_7S (7秒)
   - 按 5 键或 L1+X
   - 观察完整执行
   - 记录任何异常

2. Dance102Sar (29秒)
   - 按 U 键或 R2+X
   - 确保有足够空间
```

**验证点：**
- [ ] 技能完整执行
- [ ] 动作流畅
- [ ] 结束后能返回 LOCO 模式

#### 5.2 中等时长技能（20分钟）

```
3. GangnamStyle (32秒)
   - 按 I 键或 L2+A
   - 需要约 2m × 2m 空间

4. Dance101 (131秒)
   - 按 6 键或 L1+Y
   - 需要约 3m × 3m 空间
```

#### 5.3 长时长技能（25分钟+）

```
⚠️ 警告：这些技能需要大空间和长时间执行

5. FallAndGetup101 (168秒)
   - 按 Y 键或 R2+Y
   - 包含倒地和起身动作
   - 需要软垫保护

6. Dance204 (225秒)
   - 按 T 键或 R2+B
   - 需要 4m × 4m 空间

7. Run201 (244秒)
   - 按 9 键或 L2+X
   - 需要直线跑道 > 10m

8. Walk105 (261秒)
   - 按 P 键或 L2+Y
   - 需要直线路径 > 15m
```

**⚠️ 长技能注意事项：**
- 提前规划路径
- 确保电池电量充足（> 50%）
- 全程有人监控
- 准备随时按急停

---

## 🔍 故障排查

### 常见问题

#### 1. 连接失败

**症状：** 程序卡在 "Waiting for low state"

**解决方案：**
```bash
# 检查网络
ping <机器人IP>

# 检查网络接口配置
ifconfig

# 确认 DDS 配置
# 修改 deploy_real/config.py 中的 net 参数
```

#### 2. 控制循环超时

**症状：** 频繁出现 "control loop over time"

**原因：**
- 网络延迟过高
- CPU 负载过重
- 策略计算耗时过长

**解决方案：**
```bash
# 检查网络延迟
ping -c 100 <机器人IP>

# 检查 CPU 使用率
top

# 关闭不必要的后台程序
# 考虑降低控制频率（修改 control_dt）
# 或部署到机器人主机运行
```

#### 3. 关节抖动

**症状：** 机器人关节持续抖动

**可能原因：**
- PD 参数不匹配
- 目标位置突变
- 传感器噪声

**解决方案：**
```python
# 检查策略输出的 kp/kd 值
# 在 deploy_real.py 中添加调试输出：
print(f"KP: {kps[:5]}, KD: {kds[:5]}")

# 立即按 R 键进入 PASSIVE 模式
# 检查策略文件是否正确加载
```

#### 4. 技能无法触发

**症状：** 按键后技能不执行

**检查清单：**
```python
# 1. 确认当前状态
# 技能只能从 LOCO 或 PASSIVE 模式触发

# 2. 检查按键是否正确
# 键盘：确认没有按错键
# 手柄：确认组合键正确（如 L1+X）

# 3. 查看终端输出
# 是否有错误信息

# 4. 验证策略文件存在
ls policy/beyond_mimic_motions/*/policy.onnx
```

## 创建虚拟环境
### 使用 venv创建虚拟环境
```bash
# SSH 到机器人
ssh unitree@<机器人IP>

# 创建虚拟环境
cd ~/RoboMimic_Deploy
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install numpy pynput onnxruntime

# 验证安装
python -c "import numpy; import onnxruntime; print('OK')"

# 运行程序
python deploy_real/deploy_real.py
```
### conda
```bash
# 创建环境
conda create -n robot python=3.8
conda activate robot

# 安装依赖
conda install numpy
pip install pynput onnxruntime unitree_sdk2py

# 运行
python deploy_real/deploy_real.py
```
