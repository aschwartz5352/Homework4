import pygame
import numpy as np

pygame.init()


fpsClock = pygame.time.Clock()

SCREEN_WIDTH, SCREEN_HEIGHT = 360, 360
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill((255, 255, 255))
clock = pygame.time.Clock()

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
state = (0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)
print(state)

def getRandomXVelocity(vx):
    paddleXRandRange = (-0.015,0.015)
    newV = -vx +(2*0.015)*np.random.rand()-0.015
    if newV > 1:
        return 1
    else:
        return newV

def getRandomYVelocity(vy):
    paddleYRandRange = (-0.03,0.03)
    newV = -vy + (2*0.03)*np.random.rand()-0.03

    if newV > 1:
        return 1
    else:
        return newV
# paddleYRandRange = (-0.03,0.03)
# print((paddleXRandRange[1]-paddleXRandRange[0])*np.random.rand()+paddleXRandRange[0])
time = 0
while True:
    time += 1
    if time > 1000:
        pygame.quit()
        break
    surface.fill((255, 255, 255))


    r = pygame.Rect((state[0]*SCREEN_WIDTH, state[1]*SCREEN_HEIGHT), (10, 10))
    pygame.draw.rect(surface, (0,0,0), r)

    r = pygame.Rect((SCREEN_WIDTH-10, state[4]*SCREEN_HEIGHT), (10, paddle_height))
    pygame.draw.rect(surface, (0,0,0), r)

    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    pad = state[4]
    if y < 0:
        y = -y
        vy = -vy
    if y > 1:
        y = 2-y
        vy = -vy
    if x < 0:
        x = -x
        vx = -vx
    if x >= 1:
        x = 2*(1)-x
        vx = getRandomXVelocity(vx)
        vy = getRandomYVelocity(vy)


    state = (x+vx, y+vy, vx, vy, pad)
    print(state)
    screen.blit(surface, (0, 0))

    pygame.display.flip()
    pygame.display.update()
    fpsClock.tick(22)