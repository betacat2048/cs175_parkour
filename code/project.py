import time
import torch
import random
import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Environment import Environment


class DQN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 7, dilation=2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, 5, dilation=2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 24, 5),
            torch.nn.MaxPool2d(2, 2),
        )
        self.s_ff = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
        )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(30240+32, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2*2*7),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).flatten(max(x.dim() - 3, 0))
        s = self.s_ff(s)
        return self.ff(torch.concat((x, s), dim=-1))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, prev_state, prev_speed, action, next_state, next_speed, reward):
        self.memory.append((prev_state, prev_speed, action, next_state, next_speed, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# update target_net from policy_net by tau
def soft_update(policy_net: torch.nn.Module, target_net: torch.nn.Module, tau: float = 1e-4):
    # θ′ ← τ θ + (1 −τ)θ′
    with torch.no_grad():
        policy_net_dist_state_dict = policy_net.state_dict()
        target_net_dist_state_dict = target_net.state_dict()
        for key in policy_net_dist_state_dict:
            target_net_dist_state_dict[key] = policy_net_dist_state_dict[key]*tau + target_net_dist_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_dist_state_dict)


tau = 5e-2
gamma = 0.99
batch_size = 32
training_mode = False
device = torch.device('cuda')

policy_net_a = DQN()
target_net_a = DQN()
policy_net_a.to(device)
target_net_a.to(device)
target_net_a.load_state_dict(policy_net_a.state_dict())

policy_net_b = DQN()
target_net_b = DQN()
policy_net_b.to(device)
target_net_b.to(device)
target_net_b.load_state_dict(policy_net_b.state_dict())

steps = 0
optim = torch.optim.AdamW([
        {'params': policy_net_a.parameters()},
        {'params': policy_net_b.parameters()},
    ], lr=1e-4, weight_decay=0.01, amsgrad=False)
criterion = torch.nn.SmoothL1Loss()
memory = ReplayMemory(32*batch_size)

# saved_model = 'E:/checkpoints/Mar22_1351.pth'
saved_model = 'E:/checkpoints/Mar23_1428.pth'
# saved_model = None
if saved_model is not None:
    saved_model = torch.load(saved_model)
    policy_net_a.load_state_dict(saved_model['policy_net_a'])
    policy_net_b.load_state_dict(saved_model['policy_net_b'])
    target_net_a.load_state_dict(saved_model['target_net_a'])
    target_net_b.load_state_dict(saved_model['target_net_b'])
    optim.load_state_dict(saved_model['optim'])
    steps = saved_model['steps']

plt.ion()
plt.figure(figsize=(10, 6))
plot_labels = [f'{("-", "P")[a // 7 // 2]}{("-", "J")[(a // 7) % 2]}{("⇚", "⇐", "←", "↑", "→", "⇒", "⇛")[a % 7]}' for a in range(2*2*7)]


def select_action(state: torch.Tensor, speed: torch.Tensor, eps_threshold: float):
    with torch.inference_mode():
        policy_net_a.eval()
        policy_net_b.eval()
        if training_mode:
            Qs = policy_net_a(state[None, ...], speed[None, ...])[0] if random.random() < 0.5 else policy_net_b(state[None, ...], speed[None, ...])[0]
        else:
            Qs = (policy_net_a(state[None, ...], speed[None, ...])[0] + policy_net_b(state[None, ...], speed[None, ...])[0]) / 2
        best_action = torch.argmax(Qs, dim=-1).item()
    # action = best_action if random.random() > eps_threshold else (10 * int(random.random() < 0.25) + 5 * int(random.random() < 0.5) + np.random.choice(range(5), p=[0.175, 0.20, 0.25, 0.20, 0.175]))
    action = best_action if random.random() > eps_threshold else torch.argmax(Qs + torch.randn(28, device=device) * torch.std(Qs) * eps_threshold * 1.5, dim=-1).item()
    if random.random() < 0.2:
        print('?')
        print("?")
    plt.clf()
    plt.gca().bar(plot_labels, Qs.numpy(force=True), color=['red' if i == action else ('yellow' if i == best_action else 'blue') for i in range(2*2*7)])
    plt.gca().set_title('Q-values')
    plt.draw()
    plt.pause(0.0001)
    return action


traveled = []
durations = []
longest_time = 0
init_time = time.time()
best_time = time.time()
env = Environment(speed=1.0, size=(960, 720))
for i_episode in range(100):
    duration = 0
    total_reward = 0
    start_time = time.time()
    prev_state, prev_speed, prev_action = None, None, None
    while True:
        prev_time = time.time()
        state, reward, speed = env.get_state()
        speed = torch.as_tensor(speed.astype(np.float32)).to(device)

        if state is not None:
            state = torch.nn.functional.interpolate(torch.mean(torch.from_numpy(state).to(device).float() / 255., dim=-1)[None, None, :, :], size=(288, 384), mode='bilinear')[0, :, :, :]
            eps_threshold = 0.05 + 0.9 * np.exp(-1e-5*steps) if training_mode else 0
            action = select_action(state, speed, eps_threshold)
            env.take_action(action)

        if prev_state is not None and training_mode:
            memory.append(prev_state, prev_speed, torch.tensor(prev_action, dtype=torch.int64).to(device), state, speed, torch.tensor(reward).to(device))
        prev_state, prev_speed, prev_action = state, speed, action

        steps += 1
        duration += 1
        total_reward = total_reward * gamma + reward
        with torch.inference_mode():
            policy_net_a.eval()
            policy_net_b.eval()
            print(f'episode = {i_episode:4d}, steps = {steps:8d}, duration = {duration:3d}, total_reward = {total_reward:8.2f}, reward = {reward:6.2f}, ' 
                  f'action = {("-", "P")[action // 7 // 2]}{("-", "J")[(action // 7) % 2]}{("⇚", "⇐", "←", "↑", "→", "⇒", "⇛")[action % 7]}, '
                  f'Q = {(policy_net_a(state[None, ...], speed[None, ...])[0, action].item() + policy_net_b(state[None, ...], speed[None, ...])[0, action].item()) / 2 if state is not None else np.nan:+6.2f}; ',
                  end=''
                  )

        if state is None:
            print()
            break

        # DQN training:
        if len(memory) < batch_size or not training_mode:
            print()
            continue

        batch_prev_state, batch_prev_speed, batch_prev_action, batch_next_state, batch_next_speed, batch_reward = tuple(zip(*memory.sample(batch_size)))
        batch_reward = torch.stack(batch_reward)
        batch_prev_state = torch.stack(batch_prev_state)
        batch_prev_speed = torch.stack(batch_prev_speed)
        batch_next_speed = torch.stack(batch_next_speed)
        batch_prev_action = torch.stack(batch_prev_action)
        batch_not_final_mask = torch.tensor([s is not None for s in batch_next_state])
        batch_not_final_next_state = torch.stack([s for s in batch_next_state if s is not None])

        next_Q_value = torch.zeros(batch_size, device=device)
        idx = torch.randn(batch_size, device=device) > 0
        with torch.no_grad():
            target_net_a.eval()
            target_net_b.eval()

            next_Q_value_a = target_net_a(batch_not_final_next_state, batch_next_speed[batch_not_final_mask])
            next_Q_value_b = target_net_b(batch_not_final_next_state, batch_next_speed[batch_not_final_mask])

            next_action_a = torch.argmax(next_Q_value_a, dim=-1)
            next_action_b = torch.argmax(next_Q_value_b, dim=-1)

            next_action = torch.empty(len(next_action_a), dtype=torch.int64, device=device)
            next_action[idx[batch_not_final_mask]] = next_action_a[idx[batch_not_final_mask]]
            next_action[~idx[batch_not_final_mask]] = next_action_b[~idx[batch_not_final_mask]]

            next_Q_value_a = next_Q_value_a.gather(1, next_action[:, None]).squeeze()
            next_Q_value_b = next_Q_value_b.gather(1, next_action[:, None]).squeeze()

            next_Q_value[batch_not_final_mask] = torch.minimum(next_Q_value_a, next_Q_value_b)
        expect_Q_value = batch_reward + gamma * next_Q_value

        policy_net_a.train()
        policy_net_b.train()
        optim.zero_grad()
        loss_a = criterion(policy_net_a(batch_prev_state, batch_prev_speed).gather(1, batch_prev_action[:, None]).squeeze()[idx], expect_Q_value[idx])
        loss_b = criterion(policy_net_b(batch_prev_state, batch_prev_speed).gather(1, batch_prev_action[:, None]).squeeze()[~idx], expect_Q_value[~idx])
        loss = loss_a + loss_b
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net_a.parameters(), 1)
        torch.nn.utils.clip_grad_value_(policy_net_b.parameters(), 1)
        optim.step()

        soft_update(policy_net_a, target_net_a, tau)
        soft_update(policy_net_b, target_net_b, tau)

        print(f'loss = {loss.item(): 8.4f}, eps = {eps_threshold: 6.4f}, used: {time.time() - prev_time:4.2f}s')

    traveled.append(env.traveled)
    durations.append((time.time() - start_time))
    if durations[-1] > longest_time:
        longest_time = durations[-1]
        best_time = start_time

    if (i_episode + 1) % 100 == 0 and training_mode:
        torch.save({
            'target_net_a': target_net_a.state_dict(),
            'policy_net_a': policy_net_a.state_dict(),
            'target_net_b': target_net_b.state_dict(),
            'policy_net_b': policy_net_b.state_dict(),
            'optim': optim.state_dict(),
            'steps': steps
        }, f'E:/checkpoints/{datetime.datetime.now().strftime("%b%d_%H%M")}.pth')

    env.restart()

print(traveled)
print(durations)
print(best_time - init_time)
