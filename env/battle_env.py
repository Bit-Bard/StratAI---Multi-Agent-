import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BattleEnv(gym.Env):

    def __init__(self):
        super(BattleEnv, self).__init__()

        self.max_steps = 200
        self.current_step = 0
        self.grid_size = 10

        self.attack_cooldown_A = 0
        self.attack_cooldown_B = 0

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._last_action = 0

    # ---------------- RESET ----------------
    def reset(self):
        self.current_step = 0
        self._last_action = 0
        self._visit_counts = {}

        self.attack_cooldown_A = 0
        self.attack_cooldown_B = 0

        while True:
            self.agent1_pos = [np.random.randint(0, self.grid_size),
                               np.random.randint(0, self.grid_size)]
            self.agent2_pos = [np.random.randint(0, self.grid_size),
                               np.random.randint(0, self.grid_size)]

            dist = abs(self.agent1_pos[0] - self.agent2_pos[0]) + \
                   abs(self.agent1_pos[1] - self.agent2_pos[1])

            if dist >= 4:
                break

        self.agent1_hp = 10
        self.agent2_hp = 10

        return self._get_obs()

    # ---------------- OBS ----------------
    def _get_obs(self):
        dx = (self.agent2_pos[0] - self.agent1_pos[0]) / self.grid_size
        dy = (self.agent2_pos[1] - self.agent1_pos[1]) / self.grid_size

        return np.array([
            self.agent1_pos[0] / self.grid_size,
            self.agent1_pos[1] / self.grid_size,
            dx,
            dy,
            self.agent1_hp / 10.0,
            self.agent2_hp / 10.0,
            self._last_action / 6.0,
            1.0 - self.current_step / self.max_steps
        ], dtype=np.float32)

    # ---------------- ENEMY POLICY ----------------
    def _enemy_policy(self):
        dx = self.agent1_pos[0] - self.agent2_pos[0]
        dy = self.agent1_pos[1] - self.agent2_pos[1]

        if self._is_adjacent():
            if self.attack_cooldown_B == 0 and np.random.rand() < 0.7:
                return 4
            # on cooldown or 30% chance: reposition instead of falling through
            if abs(dx) >= abs(dy):
                return 2 if dx > 0 else 3
            else:
                return 0 if dy > 0 else 1

        # not adjacent: chase with 30% noise (movement only)
        if np.random.rand() < 0.3:
            return np.random.randint(0, 4)

        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 1 if dy > 0 else 0

    # ---------------- STEP ----------------
    def step(self, action1, action2):

        self.current_step += 1
        self._last_action = action1

        reward1 = 0.0
        reward2 = 0.0

        was_adjacent = self._is_adjacent()
        old_dist = self._manhattan()

        old_pos1 = self.agent1_pos.copy()
        old_pos2 = self.agent2_pos.copy()

        # =========================
        # MOVEMENT
        # =========================
        if action1 in [0,1,2,3]:
            self._move(self.agent1_pos, action1)

        if action2 in [0,1,2,3]:
            self._move(self.agent2_pos, action2)

        if self.agent1_pos == self.agent2_pos:
            self.agent1_pos = old_pos1
            self.agent2_pos = old_pos2

        # =========================
        # DODGE FLAGS
        # =========================
        dodge_A = (action1 == 6)
        dodge_B = (action2 == 6)

        # =========================
        # COMBAT (SEQUENTIAL)
        # =========================
        if was_adjacent:

            if np.random.rand() < 0.5:
                order = ["A", "B"]
            else:
                order = ["B", "A"]

            for attacker in order:

                # ---------- A attacks ----------
                if attacker == "A":
                    if action1 == 4 and self.attack_cooldown_A == 0:
                        if dodge_B and np.random.rand() < 0.7:
                            reward1 -= 2
                            reward2 += 2
                        else:
                            if np.random.rand() < 0.2:
                                damage = 4
                                reward1 += 2
                            else:
                                damage = np.random.randint(1, 3)
                            self.agent2_hp -= damage
                            reward1 += 5.5
                            self.attack_cooldown_A = 2
                            if self.agent2_hp <= 0:
                                break
                    elif action1 == 4:
                        reward1 -= 3

                # ---------- B attacks ----------
                else:
                    if action2 == 4 and self.attack_cooldown_B == 0:
                        if dodge_A and np.random.rand() < 0.7:
                            reward2 -= 2
                            reward1 += 2
                        else:
                            if np.random.rand() < 0.2:
                                damage = 4
                                reward2 += 2
                            else:
                                damage = np.random.randint(1, 3)
                            self.agent1_hp -= damage
                            reward2 += 5
                            self.attack_cooldown_B = 2
                            if self.agent1_hp <= 0:
                                break
                    elif action2 == 4:
                        reward2 -= 3

        # =========================
        # DISTANCE REWARD
        # =========================
        new_dist = self._manhattan()

        if not was_adjacent:
            delta = old_dist - new_dist
            if delta > 0:
                reward1 += 2
                reward2 += 2
            else:
                reward1 -= 1
                reward2 -= 1

        # =========================
        # DODGE REWARD
        # =========================
        if dodge_A:
            reward1 += 0.5
        if dodge_B:
            reward2 += 0.5

        # =========================
        # SURVIVAL REWARD
        # =========================
        reward1 += (self.agent1_hp - self.agent2_hp) * 0.1
        reward2 += (self.agent2_hp - self.agent1_hp) * 0.1

        # =========================
        # STEP PENALTY
        # =========================
        reward1 -= 0.05
        reward2 -= 0.05

        # =========================
        # COOLDOWN
        # =========================
        self.attack_cooldown_A = max(0, self.attack_cooldown_A - 1)
        self.attack_cooldown_B = max(0, self.attack_cooldown_B - 1)

        # =========================
        # DONE
        # =========================
        done = False

        if self.agent2_hp <= 0:
            reward1 += 20
            reward2 -= 20
            done = True

        if self.agent1_hp <= 0:
            reward1 -= 20
            reward2 += 20
            done = True

        if self.current_step >= self.max_steps:
            hp_diff = self.agent1_hp - self.agent2_hp
            reward1 += hp_diff * 1.0
            reward2 -= hp_diff * 1.0
            done = True

        return self._get_obs(), (reward1, reward2), done, {}

    # ---------------- MOVE ----------------
    def _move(self, pos, action):
        if action == 0:   pos[1] -= 1
        elif action == 1: pos[1] += 1
        elif action == 2: pos[0] -= 1
        elif action == 3: pos[0] += 1

        pos[0] = int(np.clip(pos[0], 0, self.grid_size - 1))
        pos[1] = int(np.clip(pos[1], 0, self.grid_size - 1))

    def _is_adjacent(self):
        return self._manhattan() == 1

    def _manhattan(self):
        return abs(self.agent1_pos[0] - self.agent2_pos[0]) + \
               abs(self.agent1_pos[1] - self.agent2_pos[1])