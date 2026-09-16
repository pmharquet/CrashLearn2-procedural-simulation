"""Capture unfiltered seed sequences with the real wide-view renderer."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import pygame
from procedural_simulation import ProceduralSimulation
from procedural_ui import RaceRenderer

def capture(seeds,output):
    pygame.init()
    output.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        sim = ProceduralSimulation(seed=seed,num_cars=1,weather='original')
        screen = pygame.Surface((1440,960))
        ui = dict(follow=0,camera_yaw=0.,overview=True,lidar=False,trails=False,paused=False,
                  speed=1,actual_speed=1.,fps=0.,replay=False,recording=False,message='')
        RaceRenderer(screen).draw(sim.snapshot(),ui)
        pygame.image.save(screen,str(output/f'seed-{seed}.png'))
    pygame.quit()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(6)))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    capture(args.seeds,args.output)
