import torch
import torch.optim as optim
import numpy as np

from env.battle_env import BattleEnv
from model.dqn import DQN
from utils.replay_buffer import ReplayBuffer

env = BattleEnv()

state_dim = 8
action_dim = 7

# Two agents
model_A = DQN(state_dim, action_dim)
model_B = DQN(state_dim, action_dim)

target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(model_A.state_dict())

optimizer = optim.Adam(model_A.parameters(), lr=0.0003)

buffer = ReplayBuffer()

gamma = 0.99
batch_size = 64

epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05

episodes = 1500

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        # ✅ Agent A (learning)
        if np.random.rand() < epsilon:
            action1 = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action1 = torch.argmax(model_A(state_tensor)).item()

        # ✅ Agent B (frozen opponent)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if np.random.rand() < 0.2:
                action2 = env.action_space.sample()
            else:
                action2 = torch.argmax(model_B(state_tensor)).item()

        # ✅ NEW step format
        next_state, (r1, r2), done, _ = env.step(action1, action2)

        # train ONLY A
        buffer.add(state, action1, r1, next_state, done)

        state = next_state
        total_reward += r1

        # ---- Training ----
        if buffer.size() < 1000:
            continue

        if buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states      = torch.FloatTensor(states)
            actions     = torch.LongTensor(actions)
            rewards     = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones       = torch.FloatTensor(dones)

            q_values      = model_A(states)
            next_q_values = target_model(next_states)

            q_value    = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            max_next_q = next_q_values.max(1)[0]
            target     = rewards + gamma * max_next_q * (1 - dones)

            loss = torch.mean((q_value - target.detach()) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), max_norm=1.0)
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # target update
    if episode % 50 == 0:
        target_model.load_state_dict(model_A.state_dict())

    # 🔥 SELF-PLAY UPDATE
    if episode % 300 == 0:
        model_B.load_state_dict(model_A.state_dict())

    if episode % 10 == 0:
        print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | Epsilon: {epsilon:.3f}")

    if episode % 200 == 0:
        torch.save(model_A.state_dict(), "model.pth")
        print(f">>> Model saved at episode {episode}")