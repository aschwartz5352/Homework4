import pygame
import numpy as np
import pickle
from matplotlib import pyplot as plt

numGames = 10
epsilon = 0.05
alpha = 0.2
gamma = 0.5
C = 10000
exploitationThreashold = 4.8

actionOptions = (0,0.04,-0.04)

np.random.seed(0)

def getRandomXVelocity(vel_x):
    # xRandRange = (-0.015,0.015)
    new_v = -vel_x + (2 * 0.015) * np.random.rand() - 0.015
    # print(vel_x, new_v)
    if new_v > 1:
        return 1
    elif abs(new_v) <= 0.03:
        return -0.03
    else:
        return new_v


def getRandomYVelocity(vel_y):
    # yRandRange = (-0.03,0.03)
    new_v = vel_y + (2 * 0.03) * np.random.rand() - 0.03

    if new_v > 1:
        return 1
    else:
        return new_v

optionsTable = {}
def Q(state):
    if state not in optionsTable:
        optionsTable[state] = [0,0,0,0]
    result = optionsTable[state]

    return result

def QSet(state, action, value):
    optionsTable[state][action] = value
    optionsTable[state][3] += 1

def f(state):
    actionVals = Q(state)

    if np.random.rand() < epsilon:
        r = np.random.randint(0,2)
        # print(actionVals)
        return actionVals[r], r, actionVals[3]
    # print(actionVals, np.amax(actionVals), actionVals.index(np.amax(actionVals)))
    max = np.amax(actionVals[0:-1])
    return max, actionVals.index(max), actionVals[3]

def cleanState(state):
    if state[0] > 1:
        return (12,0,0,0,0)
    x = int(np.floor(state[0]*12))
    y = int(np.floor(state[1]*12))
    vx = int(state[2]/abs(state[2]))
    vy = 0
    if abs(state[3]) >= 0.015:
        vy = int(state[3]/abs(state[3]))
    # pad = int(np.floor(12*state[4]/(1-0.2)))
    pad = int(np.floor(12*state[4]/(1-0.2)))
    return (x,y,vx,vy,pad)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
#
# def replay(history):
#     for state in history:
#         surface.fill((0, 0, 0))
#
#         # Draw Ball
#         pygame.draw.circle(surface, (255, 255, 255), (int(state[0] * SCREEN_WIDTH), int(state[1] * SCREEN_HEIGHT)), 5)
#
#         # Draw Paddle
#         r = pygame.Rect((SCREEN_WIDTH - 10, state[4] * SCREEN_HEIGHT), (10, paddle_height))
#         pygame.draw.rect(surface, (255, 255, 255), r)
#
#         screen.blit(surface, (0, 0))
#         pygame.display.flip()
#         pygame.display.update()
#         fpsClock.tick(60)

state = (0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)  # Change 0.01 to 0.005 for some fun :)
print(state)

play = False
time = 0
hits = 0
maxHits = 0
totalHits = 0
# recorder = []
while True:

    cState = cleanState(state)
    value, action, N = f(cState)

    reward = 0

    # Get Game State
    vx = state[2]
    vy = state[3]
    ball_x = state[0] + vx
    ball_y = state[1] + vy
    pad = state[4]

    pad += actionOptions[action]
    if pad > 1-0.2:
        pad = 1-0.2
    elif pad < 0:
        pad = 0


    # Position Checks
    # Hit Top Wall
    if ball_y < 0:
        ball_y = -ball_y
        vy = -vy
    # Hit Bottom Wall
    elif ball_y > 1:
        ball_y = 2 - ball_y
        vy = -vy
    # Hit Left Wall
    if ball_x < 0:
        ball_x = -ball_x
        vx = -vx
    # Next to Right Wall
    elif ball_x >= 1:
        # Hit Paddle
        if pad <= ball_y <= pad + 0.2:
            ball_x = 2 * 1 - ball_x
            vx = getRandomXVelocity(vx)
            vy = getRandomYVelocity(vy)

            hits += 1
            totalHits += 1

            reward = 1
        # Hit Wall
        else:
            hits = 0
            reward = -1

    state = (ball_x, ball_y, vx, vy, pad)

    if time % 1000 == 0:
        # Draw
        surface.fill((0, 0, 0))

        # Draw Ball
        pygame.event.get()
        pygame.draw.circle(surface, (255, 255, 255), (int(state[0]*SCREEN_WIDTH), int(state[1]*SCREEN_HEIGHT)), 5)

        # Draw Paddle
        r = pygame.Rect((SCREEN_WIDTH-10, state[4]*SCREEN_HEIGHT), (10, paddle_height))
        pygame.draw.rect(surface, (255, 255, 255), r)

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(60)

    # Velocity Calculations

    # if reward == -1:
    #     print("hi")

    nextState = cleanState(state)
    # print(state)
    #
    value2, action2, N = f(nextState)
    #
    decay = C/(C + N-1)
    # new_value = (1-stateAlpha)*value + stateAlpha * (reward + gamma * value2-value)
    new_value = value + alpha * decay * (reward + gamma * value2 - value)
    QSet(cState, action, new_value)

    # # Missed
    if reward == -1:
        recorder = []
        state = (0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)
        if time % 1000 == 0:
            print(time, totalHits/(1000), len(optionsTable))
            if totalHits/(1000) > exploitationThreashold:
                epsilon = 0
            totalHits = 0
            # alpha -= 0.005
        time += 1
        if time > 100000:
            pygame.quit()
            break
        #     return hits

    #
    # hits += reward


