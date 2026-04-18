
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from env.battle_env import BattleEnv


# =========================
# PPO Model
# =========================
class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value


# =========================
# Memory
# =========================
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()


# =========================
# Setup
# =========================
env = BattleEnv()

state_dim = 8
action_dim = 7

model_A = PPO(state_dim, action_dim)
model_B = PPO(state_dim, action_dim)

optimizer = optim.Adam(model_A.parameters(), lr=3e-4)

gamma = 0.99
eps_clip = 0.2
epochs = 4

memory = Memory()

episodes = 3000 #1800


# =========================
# Training Loop
# =========================
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)

        # ---- Agent A (learning) ----
        probs_A, _ = model_A(state_tensor)
        dist_A = torch.distributions.Categorical(probs_A)

        action1 = dist_A.sample()
        log_prob = dist_A.log_prob(action1)

        # ---- Agent B (frozen) ----
        with torch.no_grad():
            probs_B, _ = model_B(state_tensor)
            dist_B = torch.distributions.Categorical(probs_B)
            action2 = dist_B.sample()

        next_state, (r1, r2), done, _ = env.step(action1.item(), action2.item())

        memory.states.append(state)
        memory.actions.append(action1)
        memory.log_probs.append(log_prob.detach())
        memory.rewards.append(r1)
        memory.dones.append(done)

        state = next_state
        total_reward += r1

    # =========================
    # Compute Returns
    # =========================
    returns = []
    G = 0

    for r, d in zip(reversed(memory.rewards), reversed(memory.dones)):
        if d:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)

    states = torch.FloatTensor(memory.states)
    actions = torch.stack(memory.actions)
    old_log_probs = torch.stack(memory.log_probs).detach()

    # =========================
    # PPO Update
    # =========================
    for _ in range(epochs):
        probs, values = model_A(states)
        dist = torch.distributions.Categorical(probs)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        loss = (
            -torch.min(surr1, surr2).mean()
            + 0.5 * (returns - values.squeeze()).pow(2).mean()
            - 0.01 * entropy.mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    memory.clear()

    # 🔥 SELF-PLAY UPDATE
    if episode % 100 == 0:
        model_B.load_state_dict(model_A.state_dict())

    print(f"Episode {episode} | Reward: {total_reward:.2f}")

    if episode % 100 == 0:
        torch.save(model_A.state_dict(), "ppo_selfplay.pth")
        print(">>> Model Saved")


print("Training Complete")