import numpy as np

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

try:
    import gymnasium as gym
except ImportError:
    import gym

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -------------------- 1. 定义Q网络 --------------------
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

# -------------------- 2. 经验回放缓冲区 --------------------
class ReplayBuffer:
    """存储和采样经验（s, a, r, s', done）"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # 转换为torch张量
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(np.array(action)).unsqueeze(1)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done)).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -------------------- 3. DQN Agent --------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.q_net = QNetwork(state_dim, action_dim).to(device)          # 主网络
        self.target_net = QNetwork(state_dim, action_dim).to(device)     # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())          # 初始同步
        self.target_net.eval()                                            # 目标网络不参与训练

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.action_dim = action_dim
        self.epsilon = 1.0          # 初始探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99           # 折扣因子
        self.batch_size = 64
        self.update_target_freq = 100  # 每多少步更新目标网络

        self.step_counter = 0

    def select_action(self, state, eval_mode=False):
        """ε-greedy策略选择动作"""
        if eval_mode:
            # 测试时直接取最大Q值动作
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

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

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

        # 计算当前Q值
        current_q = self.q_net(state).gather(1, action)  # (batch, 1)

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1, keepdim=True)[0]
            target_q = reward + (1 - done) * self.gamma * next_q

        # 损失函数 (均方误差)
        loss = nn.MSELoss()(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def safe_make_env(env_name, render=False):
    kwargs = {}
    if render:
        kwargs['render_mode'] = 'human'
    try:
        return gym.make(env_name, **kwargs)
    except Exception:
        return gym.make(env_name)


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return next_state, reward, done, info


# -------------------- 4. 训练主循环 --------------------
def train_dqn(env_name='CartPole-v1', episodes=500):
    env = safe_make_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device)

    rewards_history = []
    for ep in range(episodes):
        state = reset_env(env)
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = step_env(env, action)
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

# -------------------- 5. 测试未训练的Agent --------------------
def test_untrained_agent(env_name='CartPole-v1', num_episodes=3):
    """演示完全随机策略的效果（未训练）"""
    print("\n========== 演示未训练Agent的效果（完全随机动作） ==========")
    env = safe_make_env(env_name, render=True)
    action_dim = env.action_space.n
    
    total_rewards = []
    for ep in range(num_episodes):
        state = reset_env(env)
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = random.randint(0, action_dim - 1)
            state, reward, done, _ = step_env(env, action)
            angle_deg = np.degrees(state[2])
            if angle_deg > 80 or angle_deg < -80:
                done = True
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        print(f"未训练 Episode {ep+1}: 总奖励 = {total_reward}, 步数 = {steps}")
    
    avg_reward = np.mean(total_rewards)
    print(f"未训练Agent（随机）平均奖励: {avg_reward:.2f}")
    print("==================================================\n")
    env.close()

# -------------------- 6. 测试训练好的Agent --------------------
def test_agent(agent, env_name='CartPole-v1', num_episodes=5):
    """演示训练好的Agent"""
    print("\n========== 演示训练后Agent的效果 ==========")
    env = safe_make_env(env_name, render=True)
    
    total_rewards = []
    for ep in range(num_episodes):
        state = reset_env(env)
        total_reward = 0
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _ = step_env(env, action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        print(f"已训练 Episode {ep+1}: 总奖励 = {total_reward}, 步数 = {steps}")
    
    avg_reward = np.mean(total_rewards)
    print(f"已训练Agent平均奖励: {avg_reward:.2f}")
    print("======================================\n")
    env.close()

if __name__ == "__main__":
    # 先演示未训练的Agent
    test_untrained_agent(num_episodes=10)
    
    # # 然后训练Agent
    # print("\n开始训练Agent...\n")
    # trained_agent, rewards = train_dqn(episodes=300)
    
    # # 最后演示训练好的Agent
    # test_agent(trained_agent, num_episodes=5)