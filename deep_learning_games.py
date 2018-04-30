import matplotlib.pyplot as plt
import numpy as np
import pygame


def AffineForward(A, W, b):
    return A@W + b


def ReLUForward(Z):
    return np.maximum(Z, 0.0)


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


def rungame(W1, W2, W3, W4, b1, b2, b3, b4):
    hits = 0
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
    paddle_height = 0.2 * SCREEN_HEIGHT
    state = (0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)  # Change 0.01 to 0.005 for some fun :)
    time = 0
    while True:
        pygame.event.get()
        # Draw
        surface.fill((0, 0, 0))

        # Draw Ball
        # r = pygame.Rect((state[0]*SCREEN_WIDTH, state[1]*SCREEN_HEIGHT), (10, 10))
        # pygame.draw.rect(surface, (0,0,0), r)
        pygame.draw.circle(surface, (255, 255, 255), (int(state[0] * SCREEN_WIDTH), int(state[1] * SCREEN_HEIGHT)), 5)

        # Draw Paddle
        r = pygame.Rect((SCREEN_WIDTH - 10, state[4] * SCREEN_HEIGHT), (10, paddle_height))
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
                hits += 1
                ball_x = 2 * 1 - ball_x
                vx = getRandomXVelocity(vx)
                vy = getRandomYVelocity(vy)
            # Hit Wall
            else:
                pygame.quit()
                return hits

        # Make Paddle Move. Have to adjust state with column mean/std from expert_policy
        Z1 = AffineForward(np.array([(ball_x-5.04928100e-01)/0.28852226, (ball_y-5.15399200e-01)/0.27917053, (vx-4.49400000e-04)/0.04633403, (vy-3.75000000e-05)/0.03042119, (pad-4.98532000e-01)/0.21827562]), W1, b1)
        A1 = ReLUForward(Z1)
        Z2 = AffineForward(A1, W2, b2)
        A2 = ReLUForward(Z2)
        Z3 = AffineForward(A2, W3, b3)
        A3 = ReLUForward(Z3)
        F = AffineForward(A3, W4, b4)
        move = np.argmax(F)
        # Paddle Up
        if move == 0:
            pad -= 0.04
        # Paddle Down
        elif move == 2:
            pad += 0.04

        # Make sure paddle in square
        if pad < 0:
            pad = 0
        if pad > 0.8:
            pad = 0.8

        # Velocity Calculations
        state = (ball_x + vx, ball_y + vy, vx, vy, pad)

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(60)


W1 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=1, delimiter=" ").reshape((5, 256))
b1 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=4, delimiter=" ")
W2 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=7, delimiter=" ").reshape((256, 256))
b2 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=10, delimiter=" ")
W3 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=13, delimiter=" ").reshape((256, 256))
b3 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=16, delimiter=" ")
W4 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=19, delimiter=" ").reshape((256, 3))
b4 = np.genfromtxt('our_policy20180429212152.txt', max_rows=1, skip_header=22, delimiter=" ")

NUM_GAMES = 200
games = [x+1 for x in range(NUM_GAMES)]
hits = []
total_hits = 0
for game in range(NUM_GAMES):
    result = rungame(W1, W2, W3, W4, b1, b2, b3, b4)
    hits.append(result)
    total_hits += result
print(total_hits/NUM_GAMES)
plt.plot(games, hits)
plt.title("Paddle Rebound Counts in " + str(200) + " Pong Games")
plt.xlabel("Game #")
plt.ylabel("Paddle Rebounds")
#plt.savefig("deep_learning_games.png")
