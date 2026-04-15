# CartPole DQN 代码详解文档

## 目录

1. [概述](#概述)
2. [文件结构概览](#文件结构概览)
3. [依赖与导入](#依赖与导入)
4. [QNetwork - Q 网络定义](#qnetwork---q-网络定义)
5. [ReplayBuffer - 经验回放](#replaybuffer---经验回放)
6. [DQNAgent - DQN 智能体](#dqnagent---dqn-智能体)
   - [动作选择：ε-greedy](#动作选择ε-greedy)
   - [训练步骤](#训练步骤)
7. [训练主循环](#训练主循环)
8. [测试函数](#测试函数)
   - [未训练 Agent 测试](#未训练-agent-测试)
   - [训练 Agent 测试](#训练-agent-测试)
9. [程序入口](#程序入口)
10. [DQN 关键概念总结](#dqn-关键概念总结)
11. [Python 语法规范与风格要点](#python-语法规范与风格要点)
12. [可改进建议](#可改进建议)

---

## 概述

此文档针对 `strong-learning/carpole.py` 文件进行详尽说明。该文件实现了一个基于 PyTorch 的经典 DQN（Deep Q-Network）算法，用于求解 OpenAI Gym 的 `CartPole-v1` 平衡杆任务。

该代码包含以下部分：

- Q 网络结构定义
- 经验回放缓冲区实现
- DQN 智能体逻辑
- 训练主循环
- 未训练与训练后 Agent 的测试函数

---

## 文件结构概览

`carpole.py` 的结构按功能可分为：

1. 导入依赖模块
2. `QNetwork` 类
3. `ReplayBuffer` 类
4. `DQNAgent` 类
5. `train_dqn()` 函数
6. `test_untrained_agent()` 函数
7. `test_agent()` 函数
8. `if __name__ == "__main__":` 程序入口

---

## 依赖与导入

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
```

- `gym`：OpenAI Gym，用于创建和交互环境。
- `numpy`：用于数值计算、数组转换和统计指标。
- `torch`：PyTorch，神经网络构建与训练框架。
- `torch.nn`：PyTorch 模块层级，定义神经网络结构。
- `torch.optim`：优化器模块，用于参数更新。
- `random`：Python 内置随机模块，用于随机探索和采样。
- `deque`：双端队列，用于实现固定容量的经验回放缓存。

---

## QNetwork - Q 网络定义

```python
class QNetwork(nn.Module):
    """一个简单的全连接网络，输入状态，输出每个动作的Q值"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 解释

- `QNetwork` 继承自 `nn.Module`，这是 PyTorch 定义模型的标准方式。
- 输入维度为 `state_dim`，表示环境状态向量的长度。
- 输出维度为 `action_dim`，对应每个动作的 Q 值。
- 网络由三层全连接层组成：
  - `fc1`：输入层 -> 隐藏层
  - `fc2`：隐藏层 -> 隐藏层
  - `fc3`：隐藏层 -> 输出层
- `forward()` 方法定义前向传播，使用 ReLU 激活函数。

### 作用

该网络用于估计当前状态下每个可选动作的 Q 值，后续 DQN 智能体将基于这些 Q 值选择动作或计算目标值。

---

## ReplayBuffer - 经验回放

```python
class ReplayBuffer:
    """存储和采样经验（s, a, r, s', done）"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(np.array(action)).unsqueeze(1)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done)).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
```

### 解释

- `ReplayBuffer` 用于存储经验元组 `(state, action, reward, next_state, done)`。
- `deque(maxlen=capacity)` 会自动丢弃最旧数据，保证缓存大小不超过 `capacity`。
- `push()` 方法向缓存中添加经验。
- `sample(batch_size)` 从缓存中随机抽取一个批次并转换为 PyTorch 张量：
  - `state`: float 张量
  - `action`: long 张量并扩展维度为 `(batch, 1)`
  - `reward`、`done`: float 张量并扩展为 `(batch, 1)`
- `__len__()` 允许使用 `len(buffer)` 获取当前缓存大小。

### 作用

经验回放用于打破时间序列相关性，使训练数据近似独立同分布，有利于 DQN 收敛和稳定。

---

## DQNAgent - DQN 智能体

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_freq = 100
        self.step_counter = 0
```

### 解释

- `q_net`：主网络，用于当前 Q 值估计并参与训练。
- `target_net`：目标网络，用于生成稳定的目标 Q 值。
- `load_state_dict()`：将主网络权重复制到目标网络，实现初始同步。
- `target_net.eval()`：设置目标网络为评估模式，不启用 dropout 或 batchnorm 训练行为。
- `optimizer`：Adam 优化器，学习率为 `1e-3`。
- `replay_buffer`：经验回放缓存，容量为 `10000`。
- `epsilon` 系列参数：用于 ε-greedy 探索策略。
- `gamma`：折扣因子。
- `batch_size`：训练批次大小。
- `update_target_freq`：目标网络同步频率。

### DQN 关键机制

- `q_net` 与 `target_net` 双网络结构是标准 DQN 的核心。
- 主网络不断更新，目标网络定期同步。
- 这样可以减少自反馈导致的训练不稳定。

---

### 动作选择ε-greedy

```python
    def select_action(self, state, eval_mode=False):
        """ε-greedy策略选择动作"""
        if eval_mode:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_t)
                action = q_values.argmax().item()
            return action
        else:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_net(state_t)
                    action = q_values.argmax().item()
                return action
```

### 解释

- `eval_mode=True`：测试模式，只选择最高 Q 值动作。
- 训练模式下：
  - 以概率 `epsilon` 随机探索动作
  - 否则选择当前估计的最优动作
- `torch.no_grad()` 用于关闭梯度计算，节省内存和计算开销。
- `.unsqueeze(0)` 将单个状态扩展为批次维度，形状从 `(state_dim,)` 变为 `(1, state_dim)`。

---

### 训练步骤

```python
    def train(self):
        """从缓冲区采样并更新Q网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        current_q = self.q_net(state).gather(1, action)

        with torch.no_grad():
            next_q = self.target_net(next_state).max(1, keepdim=True)[0]
            target_q = reward + (1 - done) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### 解释

训练包含以下步骤：

1. 检查经验缓存是否足够。
2. 从 `ReplayBuffer` 中随机采样一个批次。
3. 将数据移到当前设备（CPU 或 GPU）。
4. 计算当前 Q 值：`current_q = q_net(state).gather(1, action)`。
5. 计算目标 Q 值：
   - `next_q = target_net(next_state).max(1, keepdim=True)[0]`
   - `target_q = reward + (1 - done) * gamma * next_q`
6. 计算均方误差损失 `MSELoss(current_q, target_q)`。
7. 反向传播并更新主网络参数。
8. 每隔 `update_target_freq` 步同步目标网络。
9. 衰减探索率 `epsilon`。

### DQN 理论对应关系

- `current_q` 对应 `Q(s, a; θ)`。
- `target_q` 对应 `r + γ * max_{a'} Q_target(s', a'; θ−)`。
- 这里的 `θ` 是主网络参数，`θ−` 是目标网络参数。
- 采用 `MSE` 最小化目标与当前估计的差值。

---

## 训练主循环

```python
def train_dqn(env_name='CartPole-v1', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device)

    rewards_history = []
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            episode_reward += reward

        rewards_history.append(episode_reward)
        if (ep+1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1}, Avg Reward (last 50): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_history
```

### 解释

- 创建 Gym 环境并获取状态与动作维度。
- 选择可用设备：GPU (`cuda`) 优先，否则使用 CPU。
- 每一回合开始时重置环境：`state, _ = env.reset()`。
- 在单回合中反复执行：
  - 选择动作
  - 执行动作并从环境获取反馈
  - 存储 transition
  - 调用 `agent.train()` 训练网络
- 每 50 个回合输出一次平均奖励，便于观察训练进展。

### Gym API 说明

- `env.reset()` 返回 `(obs, info)`，适配 Gym v0.26+ 的新 API。
- `env.step(action)` 返回 `(next_state, reward, terminated, truncated, info)`。
- `done = terminated or truncated` 用于判断回合是否结束。

---

## 测试函数

### 未训练 Agent 测试

```python
def test_untrained_agent(env_name='CartPole-v1', num_episodes=3):
    """演示完全随机策略的效果（未训练）"""
    print("\n========== 演示未训练Agent的效果（完全随机动作） ==========")
    env = gym.make(env_name, render_mode='human')
    action_dim = env.action_space.n

    total_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = random.randint(0, action_dim - 1)
            state, reward, terminated, truncated, _ = env.step(action)
            angle_deg = np.degrees(state[2])
            if(angle_deg > 80 or angle_deg < -80):
                done = terminated or truncated
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        print(f"未训练 Episode {ep+1}: 总奖励 = {total_reward}, 步数 = {steps}")

    avg_reward = np.mean(total_rewards)
    print(f"未训练Agent（随机）平均奖励: {avg_reward:.2f}")
    print("==================================================\n")
    env.close()
```

### 解释

- 用完全随机动作演示 Actor 在 CartPole 环境中的表现。
- 该函数不使用任何深度学习模型。
- 通过 `render_mode='human'` 可视化环境。
- 在循环中根据杆角度判断回合是否结束。
- 输出每个回合及平均奖励。

---

### 训练 Agent 测试

```python
def test_agent(agent, env_name='CartPole-v1', num_episodes=5):
    """演示训练好的Agent"""
    print("\n========== 演示训练后Agent的效果 ==========")
    env = gym.make(env_name, render_mode='human')

    total_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        print(f"已训练 Episode {ep+1}: 总奖励 = {total_reward}, 步数 = {steps}")

    avg_reward = np.mean(total_rewards)
    print(f"已训练Agent平均奖励: {avg_reward:.2f}")
    print("======================================\n")
    env.close()
```

### 解释

- 使用训练好的 `agent`，在测试模式下只选择最大 Q 值动作。
- 输出每个回合的总奖励与步数。
- 可用于验证训练效果。

---

## 程序入口

```python
if __name__ == "__main__":
    test_untrained_agent(num_episodes=10)

    # # 然后训练Agent
    # print("\n开始训练Agent...\n")
    # trained_agent, rewards = train_dqn(episodes=300)

    # # 最后演示训练好的Agent
    # test_agent(trained_agent, num_episodes=5)
```

### 解释

- `if __name__ == "__main__":` 是 Python 脚本入口的标准写法。
- 直接运行 `python carpole.py` 时，执行其中代码。
- 当前代码仅演示随机未训练 Agent，训练代码已被注释。

---

## DQN 关键概念总结

1. **Q 函数**：`Q(s, a)` 表示在状态 `s` 下执行动作 `a` 的期望累计回报。
2. **ε-greedy 策略**：在训练阶段以概率 `ε` 随机探索，以概率 `1-ε` 利用当前策略。
3. **经验回放**：缓存历史 transition，并随机采样以降低样本相关性。
4. **目标网络**：使用一个延迟更新的目标网络来计算目标 Q 值，提升训练稳定性。
5. **Bellman 目标**：目标值 `r + γ * max_{a'} Q_target(s', a')`。
6. **损失函数**：最小化当前 Q 值与目标 Q 值之间的均方差。

---

## Python 语法规范与风格要点

- `import` 语句放在文件顶部。
- 类名使用 `PascalCase`，函数与变量使用 `snake_case`。
- 使用 4 个空格缩进，不要使用 Tab。
- 使用 `"""` 书写类与函数文档字符串。
- 推荐保持注释简洁、语义明确。
- `if __name__ == "__main__":` 是 Python 模块直接运行入口。
- `with torch.no_grad():` 表示不计算梯度，适合评估阶段。
- `self.target_net.eval()` 用于评估状态，避免训练特定层行为变化。

---

## 可改进建议

- 添加类型注解，如 `def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:`。
- 将各项超参数提取为构造函数参数，便于调参和复用。
- 训练阶段可以加入 `reward` 标准化、梯度裁剪、学习率调度等机制。
- 将 `train_dqn()` 返回结果绘图展示训练曲线。
- 增加 `Double DQN` 或 `Dueling DQN` 以降低过估计偏差。
- 增加 `Prioritized Replay` 提高重要样本采样效率。

---

## 使用说明

在命令行执行：

```bash
python strong-learning/carpole.py
```

若希望训练并测试训练后的模型，请取消注释 `if __name__ == "__main__"` 中的训练与测试代码块。
