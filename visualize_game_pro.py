"""
visualize_game_pro.py  —  BATTLE ARENA  ⚔
Cinematic dark sci-fi visualizer for the PPO self-play battle environment.
"""

import pygame
import torch
import torch.nn as nn
import random
import math
import time

from env.battle_env import BattleEnv

# ─────────────────────────────────────────────
#  PPO MODEL (must match training)
# ─────────────────────────────────────────────
class PPO(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(s, 128), nn.ReLU(),
            nn.Linear(128, a), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(s, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.actor(x), self.critic(x)


# ─────────────────────────────────────────────
#  CONSTANTS & PALETTE
# ─────────────────────────────────────────────
CELL        = 64
GRID        = 10
PANEL_W     = 280          # right-side info panel
GRID_PX     = GRID * CELL
WIDTH       = GRID_PX + PANEL_W
HEIGHT      = GRID_PX + 60   # bottom bar
FPS         = 5

# Colours
C_BG        = (8,   10,  18)
C_GRID      = (22,  28,  48)
C_GRID_LINE = (35,  45,  75)
C_PANEL     = (12,  15,  28)
C_PANEL_BDR = (40,  60, 120)

# Agent A  — electric cyan
CA_BASE     = (0,   210, 255)
CA_GLOW     = (0,   120, 200)
CA_CRIT     = (255, 255, 100)
CA_TRAIL    = (0,    80, 140)

# Agent B  — blood crimson
CB_BASE     = (255,  50,  70)
CB_GLOW     = (180,  20,  40)
CB_CRIT     = (255, 200,  50)
CB_TRAIL    = (120,  10,  20)

C_HP_GREEN  = (50,  230,  90)
C_HP_YELLOW = (240, 190,  30)
C_HP_RED    = (255,  50,  50)
C_WHITE     = (230, 235, 255)
C_DIM       = (80,   90, 130)
C_GOLD      = (255, 210,  50)
C_SILVER    = (180, 190, 220)

ACTION_NAMES = {
    0: "MOVE UP", 1: "MOVE DOWN", 2: "MOVE LEFT",
    3: "MOVE RIGHT", 4: "ATTACK ⚔", 5: "WAIT",  6: "DODGE ◈"
}


# ─────────────────────────────────────────────
#  PARTICLE SYSTEM
# ─────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, vx, vy, life, size=4):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.color       = color
        self.life        = life
        self.max_life    = life
        self.size        = size

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.15   # gravity
        self.vx *= 0.92
        self.life -= 1

    def draw(self, surf):
        alpha = self.life / self.max_life
        r, g, b = self.color
        col = (int(r * alpha), int(g * alpha), int(b * alpha))
        sz  = max(1, int(self.size * alpha))
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), sz)


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_hit(self, cx, cy, color, count=20):
        for _ in range(count):
            angle  = random.uniform(0, 2 * math.pi)
            speed  = random.uniform(1.5, 5.0)
            life   = random.randint(18, 35)
            size   = random.randint(3, 7)
            self.particles.append(Particle(
                cx, cy, color,
                math.cos(angle) * speed,
                math.sin(angle) * speed - 1.5,
                life, size
            ))

    def spawn_crit(self, cx, cy, color, count=40):
        for _ in range(count):
            angle  = random.uniform(0, 2 * math.pi)
            speed  = random.uniform(3.0, 9.0)
            life   = random.randint(25, 50)
            size   = random.randint(4, 10)
            self.particles.append(Particle(
                cx, cy, color,
                math.cos(angle) * speed,
                math.sin(angle) * speed - 2.0,
                life, size
            ))

    def update_and_draw(self, surf):
        alive = []
        for p in self.particles:
            p.update()
            if p.life > 0:
                p.draw(surf)
                alive.append(p)
        self.particles = alive


# ─────────────────────────────────────────────
#  TRAIL SYSTEM
# ─────────────────────────────────────────────
class Trail:
    def __init__(self, color, maxlen=6):
        self.color  = color
        self.maxlen = maxlen
        self.points = []   # list of (cx, cy)

    def add(self, cx, cy):
        self.points.append((cx, cy))
        if len(self.points) > self.maxlen:
            self.points.pop(0)

    def draw(self, surf):
        for i, (px, py) in enumerate(self.points):
            alpha = (i + 1) / len(self.points) if self.points else 0
            r, g, b = self.color
            col = (int(r * alpha * 0.5), int(g * alpha * 0.5), int(b * alpha * 0.5))
            radius = max(2, int(12 * alpha))
            pygame.draw.circle(surf, col, (px, py), radius)


# ─────────────────────────────────────────────
#  SCREEN-SHAKE
# ─────────────────────────────────────────────
class ScreenShake:
    def __init__(self):
        self.intensity = 0
        self.ox, self.oy = 0, 0

    def trigger(self, strength=8):
        self.intensity = strength

    def update(self):
        if self.intensity > 0:
            self.ox = random.randint(-self.intensity, self.intensity)
            self.oy = random.randint(-self.intensity, self.intensity)
            self.intensity = max(0, self.intensity - 1)
        else:
            self.ox, self.oy = 0, 0


# ─────────────────────────────────────────────
#  FLOATING TEXT
# ─────────────────────────────────────────────
class FloatText:
    def __init__(self, text, x, y, color, size=22, life=40):
        self.text  = text
        self.x, self.y = float(x), float(y)
        self.color = color
        self.life  = life
        self.max   = life
        self.size  = size

    def update(self):
        self.y  -= 1.2
        self.life -= 1

    def draw(self, surf, font):
        alpha = self.life / self.max
        r, g, b = self.color
        col  = (int(r * alpha), int(g * alpha), int(b * alpha))
        text = font.render(self.text, True, col)
        surf.blit(text, (int(self.x) - text.get_width() // 2, int(self.y)))


# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def draw_glowing_circle(surf, color, cx, cy, radius, glow_radius, glow_alpha=60):
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    r, g, b   = color
    pygame.draw.circle(glow_surf, (r, g, b, glow_alpha), (glow_radius, glow_radius), glow_radius)
    surf.blit(glow_surf, (cx - glow_radius, cy - glow_radius))
    pygame.draw.circle(surf, color, (cx, cy), radius)


def draw_hp_bar(surf, x, y, w, h, hp, max_hp, label, font_sm, color):
    ratio  = max(0, hp / max_hp)
    bg_col = (30, 30, 50)
    pygame.draw.rect(surf, bg_col, (x, y, w, h), border_radius=4)

    if ratio > 0.6:
        bar_col = C_HP_GREEN
    elif ratio > 0.3:
        bar_col = C_HP_YELLOW
    else:
        bar_col = C_HP_RED

    filled = int(w * ratio)
    if filled > 0:
        pygame.draw.rect(surf, bar_col, (x, y, filled, h), border_radius=4)

    # Segmented ticks every 10%
    for i in range(1, 10):
        tx = x + int(w * i / 10)
        pygame.draw.line(surf, (8, 10, 18), (tx, y), (tx, y + h), 1)

    pygame.draw.rect(surf, color, (x, y, w, h), 2, border_radius=4)

    label_surf = font_sm.render(f"{label}  {hp}/{max_hp}", True, C_WHITE)
    surf.blit(label_surf, (x, y - 20))


def agent_center(pos):
    return (pos[0] * CELL + CELL // 2, pos[1] * CELL + CELL // 2)


def draw_agent(surf, pos, base_color, glow_color, flash, pulse, label, font_sm, shake):
    cx, cy = agent_center(pos)
    cx += shake.ox
    cy += shake.oy

    # Outer glow
    glow_r = 32 + int(6 * math.sin(pulse))
    draw_glowing_circle(surf, glow_color, cx, cy, 0, glow_r, glow_alpha=50)

    # Body
    body_col = (255, 255, 180) if flash else base_color
    r2 = 22 + int(3 * math.sin(pulse))
    draw_glowing_circle(surf, body_col, cx, cy, r2, r2 + 8, glow_alpha=80)

    # Inner highlight
    highlight = tuple(min(255, c + 80) for c in body_col)
    pygame.draw.circle(surf, highlight, (cx - 6, cy - 6), 6)

    # Label under agent
    lbl = font_sm.render(label, True, base_color)
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy + 28))


def draw_attack_beam(surf, pos1, pos2, color, shake):
    cx1, cy1 = agent_center(pos1)
    cx2, cy2 = agent_center(pos2)
    cx1 += shake.ox; cy1 += shake.oy
    cx2 += shake.ox; cy2 += shake.oy
    for width in [6, 3, 1]:
        pygame.draw.line(surf, color, (cx1, cy1), (cx2, cy2), width)


def draw_grid(surf):
    surf.fill(C_GRID)
    # Subtle scanlines
    for y in range(0, GRID_PX, 4):
        pygame.draw.line(surf, (0, 0, 0, 15), (0, y), (GRID_PX, y), 1)
    # Grid lines
    for i in range(GRID + 1):
        pygame.draw.line(surf, C_GRID_LINE, (i * CELL, 0), (i * CELL, GRID_PX), 1)
        pygame.draw.line(surf, C_GRID_LINE, (0, i * CELL), (GRID_PX, i * CELL), 1)
    # Corner dots
    for x in range(GRID + 1):
        for y in range(GRID + 1):
            pygame.draw.circle(surf, C_PANEL_BDR, (x * CELL, y * CELL), 2)


def draw_panel(surf, env, step, max_steps, action1, action2,
               last_event, font_title, font_med, font_sm, font_xs, px):
    """Right-side info panel."""
    panel_rect = pygame.Rect(px, 0, PANEL_W, HEIGHT)
    pygame.draw.rect(surf, C_PANEL, panel_rect)
    pygame.draw.line(surf, C_PANEL_BDR, (px, 0), (px, HEIGHT), 2)

    y = 18
    title = font_title.render("⚔  BATTLE", True, C_GOLD)
    surf.blit(title, (px + PANEL_W // 2 - title.get_width() // 2, y))
    y += 42

    sub = font_xs.render("ARENA  //  AI COMBAT SIM", True, C_DIM)
    surf.blit(sub, (px + PANEL_W // 2 - sub.get_width() // 2, y))
    y += 28

    # Divider
    pygame.draw.line(surf, C_PANEL_BDR, (px + 12, y), (px + PANEL_W - 12, y), 1)
    y += 16

    # HP bars
    draw_hp_bar(surf, px + 20, y + 20, PANEL_W - 40, 16,
                env.agent1_hp, 10, "AGENT A", font_sm, CA_BASE)
    y += 60
    draw_hp_bar(surf, px + 20, y + 20, PANEL_W - 40, 16,
                env.agent2_hp, 10, "AGENT B", font_sm, CB_BASE)
    y += 60

    pygame.draw.line(surf, C_PANEL_BDR, (px + 12, y), (px + PANEL_W - 12, y), 1)
    y += 14

    # Step counter
    step_pct = step / max_steps
    step_col = C_HP_GREEN if step_pct < 0.6 else (C_HP_YELLOW if step_pct < 0.85 else C_HP_RED)
    slbl = font_sm.render("STEP", True, C_DIM)
    surf.blit(slbl, (px + 20, y))
    sval = font_med.render(f"{step} / {max_steps}", True, step_col)
    surf.blit(sval, (px + PANEL_W - sval.get_width() - 20, y))
    y += 30

    # Timer bar
    pygame.draw.rect(surf, (30, 30, 50), (px + 20, y, PANEL_W - 40, 8), border_radius=4)
    filled = int((PANEL_W - 40) * (1 - step_pct))
    if filled > 0:
        pygame.draw.rect(surf, step_col, (px + 20, y, filled, 8), border_radius=4)
    y += 22

    pygame.draw.line(surf, C_PANEL_BDR, (px + 12, y), (px + PANEL_W - 12, y), 1)
    y += 14

    # Last actions
    a1_lbl = font_xs.render("A  LAST ACTION", True, C_DIM)
    surf.blit(a1_lbl, (px + 20, y))
    y += 18
    a1_act = font_sm.render(ACTION_NAMES.get(action1, "—"), True, CA_BASE)
    surf.blit(a1_act, (px + 20, y))
    y += 28

    a2_lbl = font_xs.render("B  LAST ACTION", True, C_DIM)
    surf.blit(a2_lbl, (px + 20, y))
    y += 18
    a2_act = font_sm.render(ACTION_NAMES.get(action2, "—"), True, CB_BASE)
    surf.blit(a2_act, (px + 20, y))
    y += 32

    pygame.draw.line(surf, C_PANEL_BDR, (px + 12, y), (px + PANEL_W - 12, y), 1)
    y += 14

    # Last event log
    ev_lbl = font_xs.render("COMBAT LOG", True, C_DIM)
    surf.blit(ev_lbl, (px + 20, y))
    y += 20
    for line in last_event[-4:]:
        txt = font_xs.render(line, True, C_WHITE)
        surf.blit(txt, (px + 20, y))
        y += 18

    # Cooldown indicators
    y += 10
    pygame.draw.line(surf, C_PANEL_BDR, (px + 12, y), (px + PANEL_W - 12, y), 1)
    y += 14
    cd_lbl = font_xs.render("ATTACK COOLDOWN", True, C_DIM)
    surf.blit(cd_lbl, (px + 20, y))
    y += 20

    for agent_label, cd, color in [("A", env.attack_cooldown_A, CA_BASE),
                                    ("B", env.attack_cooldown_B, CB_BASE)]:
        for i in range(2):
            filled_pip = i < cd
            col = color if filled_pip else (40, 50, 80)
            pygame.draw.rect(surf, col, (px + 20 + i * 30, y, 22, 12), border_radius=3)
            if not filled_pip:
                pygame.draw.rect(surf, (60, 70, 110), (px + 20 + i * 30, y, 22, 12), 1, border_radius=3)
        lbl_cd = font_xs.render(agent_label, True, color)
        surf.blit(lbl_cd, (px + 90, y))
        y += 22


def draw_end_screen(surf, winner_text, sub_text, font_big, font_med, shake):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surf.blit(overlay, (0, 0))

    cx = WIDTH // 2 + shake.ox
    cy = HEIGHT // 2 + shake.oy

    # Glow ring
    glow = pygame.Surface((300, 300), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 210, 50, 40), (150, 150), 140)
    surf.blit(glow, (cx - 150, cy - 150))

    title = font_big.render(winner_text, True, C_GOLD)
    surf.blit(title, (cx - title.get_width() // 2, cy - 60))

    sub = font_med.render(sub_text, True, C_SILVER)
    surf.blit(sub, (cx - sub.get_width() // 2, cy + 10))

    hint = font_med.render("Press  R  to restart  |  Q  to quit", True, C_DIM)
    surf.blit(hint, (cx - hint.get_width() // 2, cy + 55))


def draw_bottom_bar(surf, episode, font_xs):
    y = GRID_PX
    pygame.draw.rect(surf, C_PANEL, (0, y, GRID_PX, 60))
    pygame.draw.line(surf, C_PANEL_BDR, (0, y), (GRID_PX, y), 1)
    lbl = font_xs.render(f"EPISODE  {episode}   //   PPO SELF-PLAY   //   BATTLE ARENA v2", True, C_DIM)
    surf.blit(lbl, (GRID_PX // 2 - lbl.get_width() // 2, y + 20))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    env   = BattleEnv()
    model = PPO(8, 7)
    model.load_state_dict(torch.load("ppo_selfplay.pth", map_location="cpu"))
    model.eval()

    pygame.init()
    pygame.display.set_caption("⚔  BATTLE ARENA  —  PPO Self-Play")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()

    # Fonts (fall back gracefully if monospace not found)
    def load_font(size, bold=False):
        for name in ["Courier New", "Consolas", "DejaVu Sans Mono", "monospace"]:
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    font_title = load_font(30, bold=True)
    font_big   = load_font(48, bold=True)
    font_med   = load_font(22)
    font_sm    = load_font(17)
    font_xs    = load_font(14)

    # Systems
    particles = ParticleSystem()
    trail_A   = Trail(CA_TRAIL, maxlen=7)
    trail_B   = Trail(CB_TRAIL, maxlen=7)
    shake     = ScreenShake()
    floats    = []

    episode     = 1
    pulse       = 0.0
    beam_timer  = 0        # frames to show attack beam
    beam_color  = CA_BASE
    event_log   = ["Battle started!"]

    last_action1 = 0
    last_action2 = 0
    game_over    = False
    winner_text  = ""
    sub_text     = ""

    # Grid surface (static, drawn once)
    grid_surf = pygame.Surface((GRID_PX, GRID_PX))
    draw_grid(grid_surf)

    state = env.reset()
    prevA_hp = env.agent1_hp
    prevB_hp = env.agent2_hp

    def reset_episode():
        nonlocal state, game_over, winner_text, sub_text
        nonlocal prevA_hp, prevB_hp, episode
        nonlocal last_action1, last_action2, beam_timer
        state = env.reset()
        prevA_hp  = env.agent1_hp
        prevB_hp  = env.agent2_hp
        game_over = False
        winner_text = sub_text = ""
        trail_A.points.clear()
        trail_B.points.clear()
        particles.particles.clear()
        floats.clear()
        event_log.clear()
        event_log.append(f"Episode {episode} — FIGHT!")
        last_action1 = last_action2 = 0
        beam_timer   = 0
        episode     += 1

    running = True
    while running:

        clock.tick(FPS)
        pulse += 0.25

        # ── EVENTS ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r and game_over:
                    reset_episode()

        # ── STEP ──
        if not game_over:
            # Agent A — trained model
            with torch.no_grad():
                s     = torch.FloatTensor(state).unsqueeze(0)
                probs, _ = model(s)
                probs    = probs.squeeze(0)
                action1  = torch.argmax(probs).item()

            # Agent B — scripted enemy
            action2 = env._enemy_policy()

            state, rewards, done, _ = env.step(action1, action2)
            last_action1, last_action2 = action1, action2

            # ── Trail ──
            trail_A.add(*agent_center(env.agent1_pos))
            trail_B.add(*agent_center(env.agent2_pos))

            # ── Attack effects ──
            hit_A = env.agent2_hp < prevB_hp
            hit_B = env.agent1_hp < prevA_hp
            dmg_A = prevB_hp - env.agent2_hp
            dmg_B = prevA_hp - env.agent1_hp

            if hit_A:
                cx, cy = agent_center(env.agent2_pos)
                is_crit = dmg_A >= 4
                particles.spawn_crit(cx, cy, CA_CRIT) if is_crit else particles.spawn_hit(cx, cy, CA_BASE)
                shake.trigger(10 if is_crit else 5)
                beam_timer = 3
                beam_color = CA_BASE
                tag = f"💥 CRIT! A deals {dmg_A}" if is_crit else f"A hits B  -{dmg_A} HP"
                event_log.append(tag)
                floats.append(FloatText(
                    f"{'CRIT! ' if is_crit else ''}-{dmg_A}",
                    cx, cy - 30,
                    CA_CRIT if is_crit else CA_BASE,
                    size=28 if is_crit else 20,
                    life=45 if is_crit else 35
                ))

            if hit_B:
                cx, cy = agent_center(env.agent1_pos)
                is_crit = dmg_B >= 4
                particles.spawn_crit(cx, cy, CB_CRIT) if is_crit else particles.spawn_hit(cx, cy, CB_BASE)
                shake.trigger(10 if is_crit else 5)
                tag = f"💥 CRIT! B deals {dmg_B}" if is_crit else f"B hits A  -{dmg_B} HP"
                event_log.append(tag)
                floats.append(FloatText(
                    f"{'CRIT! ' if is_crit else ''}-{dmg_B}",
                    cx, cy - 30,
                    CB_CRIT if is_crit else CB_BASE,
                    size=28 if is_crit else 20,
                    life=45 if is_crit else 35
                ))

            if action1 == 6:
                event_log.append("A dodges!")
            if action2 == 6:
                event_log.append("B dodges!")

            event_log = event_log[-12:]  # rolling window

            prevA_hp = env.agent1_hp
            prevB_hp = env.agent2_hp

            if done:
                game_over = True
                if env.agent2_hp <= 0:
                    winner_text = "AGENT A  WINS"
                    sub_text    = f"B eliminated  |  A HP: {env.agent1_hp}/10"
                    particles.spawn_crit(GRID_PX // 2, GRID_PX // 2, CA_CRIT, count=80)
                elif env.agent1_hp <= 0:
                    winner_text = "AGENT B  WINS"
                    sub_text    = f"A eliminated  |  B HP: {env.agent2_hp}/10"
                    particles.spawn_crit(GRID_PX // 2, GRID_PX // 2, CB_CRIT, count=80)
                else:
                    # Timeout: HP winner
                    if env.agent1_hp > env.agent2_hp:
                        winner_text = "AGENT A  WINS"
                        sub_text    = f"Time out — HP advantage  {env.agent1_hp} vs {env.agent2_hp}"
                    elif env.agent2_hp > env.agent1_hp:
                        winner_text = "AGENT B  WINS"
                        sub_text    = f"Time out — HP advantage  {env.agent2_hp} vs {env.agent1_hp}"
                    else:
                        winner_text = "DRAW"
                        sub_text    = "Equal HP at time limit"
                shake.trigger(15)

        # ── UPDATE SYSTEMS ──
        shake.update()
        floats = [f for f in floats if f.life > 0]
        for f in floats:
            f.update()

        # ── DRAW ──
        screen.fill(C_BG)

        # Grid (with shake offset)
        screen.blit(grid_surf, (shake.ox, shake.oy))

        # Trails
        trail_A.draw(screen)
        trail_B.draw(screen)

        # Attack beam
        if beam_timer > 0:
            draw_attack_beam(screen, env.agent1_pos, env.agent2_pos, beam_color, shake)
            beam_timer -= 1

        # Agents
        draw_agent(screen, env.agent1_pos, CA_BASE, CA_GLOW,
                   flash=(beam_timer > 0 and beam_color == CA_BASE),
                   pulse=pulse, label="A", font_sm=font_sm, shake=shake)

        draw_agent(screen, env.agent2_pos, CB_BASE, CB_GLOW,
                   flash=(beam_timer > 0 and beam_color == CB_BASE),
                   pulse=pulse + math.pi, label="B", font_sm=font_sm, shake=shake)

        # Particles + float texts
        particles.update_and_draw(screen)
        for f in floats:
            f.draw(screen, font_sm)

        # Adjacency highlight
        if env._is_adjacent():
            cx1, cy1 = agent_center(env.agent1_pos)
            cx2, cy2 = agent_center(env.agent2_pos)
            mid = ((cx1 + cx2) // 2 + shake.ox, (cy1 + cy2) // 2 + shake.oy)
            pygame.draw.circle(screen, C_GOLD, mid, 6 + int(4 * math.sin(pulse * 2)))

        # Panel
        draw_panel(screen, env, env.current_step, env.max_steps,
                   last_action1, last_action2, event_log,
                   font_title, font_med, font_sm, font_xs, GRID_PX)

        # Bottom bar
        draw_bottom_bar(screen, episode, font_xs)

        # End screen
        if game_over:
            draw_end_screen(screen, winner_text, sub_text, font_big, font_med, shake)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()