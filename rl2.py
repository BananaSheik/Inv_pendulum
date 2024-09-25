import gym
import math
import pygame
from stable_baselines3 import DQN

pygame.init()

width, height = 800, 600
x_cart_scale, pendulum_len = 150, 200
cart_width, cart_height = 100, 5
y_cart = height - 100


scrn = pygame.display.set_mode((width, height))

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = DQN.load("dqn_cartpole")

observation, _ = env.reset()

running = True
while running:
    scrn.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)

    x_cart = int(observation[0] * x_cart_scale) + width // 2 
    ang = observation[2]

    x_p = x_cart + pendulum_len * math.sin(ang)
    y_p = y_cart - pendulum_len * math.cos(ang)

    pygame.draw.line(scrn, (0, 0, 255), (x_p, y_p), (x_cart, y_cart), 2)
    pygame.draw.circle(scrn, (0, 255, 0), (int(x_p), int(y_p)), 15)
    pygame.draw.rect(scrn, (255, 0, 0), pygame.Rect(x_cart - cart_width // 2, y_cart, cart_width, cart_height))

    pygame.display.update()

    if terminated:
        print("Pendulum exceeded 12 degrees from the vertical.")
        observation, _ = env.reset()

    pygame.time.wait(50)

env.close()
pygame.quit()
