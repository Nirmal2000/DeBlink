import pygame
import sys
from time import sleep
   
screen = pygame.display.set_mode((800,600))
done = False
face = pygame.image.load('./face.jpg')
left = pygame.image.load('./left.jpg')
right = pygame.image.load('./right.jpg')
both = pygame.image.load('./both.jpg')
ld = pygame.image.load('./lookdown.jpg')
lu = pygame.image.load('./lookup.jpg')
ll = pygame.image.load('./lookleft.jpg')
lr = pygame.image.load('./lookright.jpg')

while not done:
    close = pygame.event.wait()
    event = pygame.key.get_pressed()
    if event[pygame.K_UP]:
        screen.blit(both,(0,0))
    elif event[pygame.K_LEFT]:
        screen.blit(left,(0,0))
    elif event[pygame.K_RIGHT]:
        screen.blit(right,(0,0))
    elif event[pygame.K_w]:
        screen.blit(lu,(0,0))
    elif event[pygame.K_s]:
        screen.blit(ld,(0,0))
    elif event[pygame.K_a]:
        screen.blit(ll,(0,0))
    elif event[pygame.K_d]:
        screen.blit(lr,(0,0))
    else:
        screen.blit(face,(0,0))
    if close.type == pygame.QUIT:            
        break
    
    pygame.display.update()
pygame.quit()
sys.exit()