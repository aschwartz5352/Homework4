import pygame

pygame.init()

print("hello")

fpsClock = pygame.time.Clock()

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill((255, 255, 255))
clock = pygame.time.Clock()

pygame.key.set_repeat(1, 40)

GRIDSIZE = 10
GRID_WIDTH = SCREEN_WIDTH / GRIDSIZE
GRID_HEIGHT = SCREEN_HEIGHT / GRIDSIZE
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

screen.blit(surface, (0, 0))
time = 0
while True:
    time += 1
    if time > 255:
        pygame.quit()
        break
    surface.fill((255, 255, 255))

    r = pygame.Rect((time, 0), (GRIDSIZE, GRIDSIZE))
    pygame.draw.rect(surface, (time,0,0), r)

    # font = pygame.font.Font(None, 36)
    # text = font.render(str(snake.length), 1, (10, 10, 10))
    # textpos = text.get_rect()
    # textpos.centerx = 20
    # surface.blit(text, textpos)
    screen.blit(surface, (0, 0))

    pygame.display.flip()
    pygame.display.update()
    # fpsClock.tick(FPS + snake.length / 3)