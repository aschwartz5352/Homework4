import pygame
import numpy as np


def getRandomXVelocity(vel_x):
    # xRandRange = (-0.015,0.015)
    new_v = -vel_x + (2 * 0.015) * np.random.rand() - 0.015
    if new_v > 1:
        return 1
    elif abs(new_v) <= 0.03:
        return 0.03
    else:
        return new_v


def getRandomYVelocity(vel_y):
    # yRandRange = (-0.03,0.03)
    new_v = vel_y + (2 * 0.03) * np.random.rand() - 0.03
    if new_v > 1:
        return 1
    else:
        return new_v


pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 360, 360
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill((0, 0, 0))
fpsClock = pygame.time.Clock()

pygame.key.set_repeat(1, 40)

GRIDSIZE = 30
GRID_WIDTH = SCREEN_WIDTH / GRIDSIZE
GRID_HEIGHT = SCREEN_HEIGHT / GRIDSIZE
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

screen.blit(surface, (0, 0))

paddle_height = 0.2*SCREEN_HEIGHT
state = (0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)  # Change 0.01 to 0.005 for some fun :)
print(state)

time = 0
while True:
    time += 1
    if time > 1000:
        pygame.quit()
        break

    # Draw
    surface.fill((0, 0, 0))

    # Draw Ball
    # r = pygame.Rect((state[0]*SCREEN_WIDTH, state[1]*SCREEN_HEIGHT), (10, 10))
    # pygame.draw.rect(surface, (0,0,0), r)
    pygame.draw.circle(surface, (255, 255, 255), (int(state[0]*SCREEN_WIDTH), int(state[1]*SCREEN_HEIGHT)), 5)

    # Draw Paddle
    r = pygame.Rect((SCREEN_WIDTH-10, state[4]*SCREEN_HEIGHT), (10, paddle_height))
    pygame.draw.rect(surface, (255, 255, 255), r)

    # Get Game State
    ball_x = state[0]
    ball_y = state[1]
    vx = state[2]
    vy = state[3]
    pad = state[4]

    # Position Checks
    # Hit Top Wall
    if ball_y < 0:
        ball_y = -ball_y
        vy = -vy
    # Hit Bottom Wall
    if ball_y > 1:
        ball_y = 2 - ball_y
        vy = -vy
    # Hit Left Wall
    if ball_x < 0:
        ball_x = -ball_x
        vx = -vx
    # Next to Right Wall
    if ball_x >= 1:
        # Hit Paddle
        if pad <= ball_y <= pad + 0.2:
            ball_x = 2 * 1 - ball_x
            vx = getRandomXVelocity(vx)
            vy = getRandomYVelocity(vy)
        # Hit Wall
        else:
            # THE FOLLOWING IS TEMPORARY. HITTING THE RIGHT WALL SHOULD BE A TERMINATION STATE
            ball_x = 2 * 1 - ball_x
            vx = -vx

    # Velocity Calculations
    state = (ball_x + vx, ball_y + vy, vx, vy, pad)
    print(state)

    screen.blit(surface, (0, 0))
    pygame.display.flip()
    pygame.display.update()
    fpsClock.tick(60)
