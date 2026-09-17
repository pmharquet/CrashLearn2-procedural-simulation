"""Export the actual simulator renderer at exactly 10x playback speed."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS','permonitorv2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pygame
import numpy as np
from PIL import Image, ImageChops
from procedural_demo import drive
from procedural_simulation import ProceduralSimulation, load_agent
from procedural_ui import RaceRenderer, COLORS, TEXT, MUTED, ACCENT
from procedural_window import RaceWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent',type=Path)
    parser.add_argument('--seed',type=int,default=6)
    parser.add_argument('--output',type=Path,default=Path('docs/media/course-x10.gif'))
    parser.add_argument('--renderer',choices=('auto','gpu','software'),default='auto')
    args = parser.parse_args()
    sim = ProceduralSimulation(args.seed,num_cars=4,weather='dynamic')
    agents = [load_agent(args.agent) for _ in range(4)]
    if args.renderer == 'software': os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    pygame.init()
    frames = []
    window = None
    try:
        window = RaceWindow((1440,960),backend=args.renderer)
        screen = window.screen
        renderer = RaceRenderer(screen,window.gpu)
        ui = dict(follow=0,camera_yaw=0.,overview=False,infos=True,lidar=False,
                  trails=True,paused=False,speed=10,actual_speed=10.,fps=12.,
                  ticks_per_second=200.,ai_per_second=0.,
                  replay=True,recording=False,message='EXPORT HORS LIGNE / LECTURE x10')
        # GIF durations must be multiples of 10 ms. 80/80/90 ms gives
        # 12 frames/s; each frame advances exactly 10 times its duration.
        # Offline encoding does not claim a realtime performance measurement.
        durations = [80,80,90]*48
        for index,duration in enumerate(durations):
            pygame.event.pump()
            for _ in range(duration//5):
                if sim.terminated or sim.truncated:
                    raise RuntimeError('Race ended before the requested capture was complete')
                drive(sim,agents)
                if sim.status[ui['follow']] != 1 and not sim.terminated:
                    ui['follow'] = next(i for i in range(4) if sim.status[i] == 1)
                delta = (sim.states[ui['follow'],4]-ui['camera_yaw']+np.pi)%(2*np.pi)-np.pi
                ui['camera_yaw'] += delta*min(1.,sim.dt*4)
            if sim.status[ui['follow']] != 1:
                ui['follow'] = next(i for i in range(4) if sim.status[i] == 1)
            # Keep one continuous race: eight seconds close-up, four seconds
            # showing the streamed course, without changing the simulation.
            ui['overview'] = index>=96
            renderer._hud_key = None
            renderer.draw(sim.snapshot(),ui)
            if window.gpu:
                size = screen.get_size()
                frame = Image.frombytes('RGB',size,window.gpu.ctx.screen.read(viewport=(0,0,*size),components=3))
                frame = frame.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                frame = Image.frombytes('RGB',screen.get_size(),pygame.image.tobytes(screen,'RGB'))
            frame = frame.resize((960,640),Image.Resampling.LANCZOS)
            pygame.display.flip()
            frames.append(frame)
            if (index+1)%24 == 0:
                print(f'{index+1}/144 frames, {sim.steps*sim.dt:.0f}s simulated',flush=True)
        # Sample the whole race, including changing weather, for one stable
        # palette. Reserve exact UI/car colors even when they occupy few pixels.
        samples = Image.new('RGB',(96*12,68))
        for i in range(12):
            samples.paste(frames[i*12].resize((96,68)),(i*96,0))
        adaptive = samples.quantize(colors=118,dither=Image.Dither.NONE)
        colors = adaptive.getpalette()[:354]
        reserved = COLORS+[TEXT,MUTED,ACCENT,(255,202,103),(247,192,88),(222,99,114)]
        colors += [value for color in reserved for value in color]
        palette = Image.new('P',(1,1))
        palette.putpalette(colors+[0]*(768-len(colors)))
        for i,frame in enumerate(frames):
            frames[i] = frame.quantize(palette=palette,dither=Image.Dither.NONE)
            frame.close()
        samples.close(); adaptive.close(); palette.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,
                       loop=0,optimize=True,disposal=1)
        with Image.open(args.output) as saved:
            if saved.n_frames != len(frames) or saved.info.get('loop') != 0:
                raise RuntimeError('GIF frame count or loop flag changed')
            total_duration = 0
            for i,frame in enumerate(frames):
                saved.seek(i)
                total_duration += saved.info['duration']
                if ImageChops.difference(saved.convert('RGB'),frame.convert('RGB')).getbbox():
                    raise RuntimeError(f'GIF encoding changed frame {i}')
            if total_duration != sum(durations):
                raise RuntimeError('GIF playback duration changed')
        print(f'{args.output}: {args.output.stat().st_size/1024/1024:.2f} MiB',flush=True)
    finally:
        if window and window.gpu: window.gpu.release()
        pygame.quit()
        for frame in frames: frame.close()


if __name__ == '__main__':
    main()
