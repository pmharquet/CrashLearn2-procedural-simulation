"""Live procedural race, streaming recordings and interactive replay."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from procedural_simulation import ProceduralSimulation, load_agent, WEATHER_MODES, ROAD_PROFILES
from procedural_recording import RaceRecorder, RaceReplay


def make_agents(args):
    agents = [load_agent(args.agent) for _ in range(args.cars)]
    if args.ppo:
        from stable_baselines3 import PPO
        from procedural_train import ProceduralEnv
        class TrainedAgent:
            def __init__(self):
                self.model = PPO.load(args.ppo)
                size = self.model.observation_space.shape[0]
                if size not in (102,131):
                    raise ValueError('Unsupported PPO observation schema')
                self.legacy = size == 102
            def predict(self,obs,info):
                action,_ = self.model.predict(ProceduralEnv.encode(obs,legacy=self.legacy),deterministic=True)
                return float(4+6*action[0]),float(.4189*action[1])
        agents[0] = TrainedAgent()
    return agents


def drive(sim,agents):
    info = sim.info()
    actions = [agent.predict(sim.observation(i),info) if sim.status[i] == 1 else (0.,0.)
               for i,agent in enumerate(agents)]
    return sim.step(actions)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent',type=Path)
    parser.add_argument('--ppo',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cars',type=int,choices=range(1,5),default=4)
    parser.add_argument('--weather',choices=WEATHER_MODES,default='cycle')
    parser.add_argument('--profile',choices=ROAD_PROFILES,default='mixed')
    parser.add_argument('--finish-distance',type=float)
    parser.add_argument('--speed',type=int,choices=range(1,11),default=3)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--fullscreen',action='store_true')
    parser.add_argument('--overview',action='store_true')
    parser.add_argument('--window-size',nargs=2,type=int,metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--frames',type=int,default=0,help='Close after N rendered frames (QA)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--record',type=Path)
    group.add_argument('--replay',type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    replay = RaceReplay(args.replay) if args.replay else None
    recorder = None
    sim = None
    agents = None
    try:
        if not replay:
            sim = ProceduralSimulation(args.seed,num_cars=args.cars,weather=args.weather,
                                       profile=args.profile,finish_distance=args.finish_distance)
            agents = make_agents(args)
            if args.record:
                recorder = RaceRecorder(args.record)
                recorder.append(sim.snapshot())
        if args.headless:
            if replay:
                print(json.dumps(dict(frames=replay.count,first=replay.frame(0)['time'],
                                      last=replay.frame(replay.count-1)['time'])))
                return
            courses = 1
            contacts = 0
            distance = 0.
            started = time.perf_counter()
            for k in range(args.steps):
                if sim.terminated or sim.truncated:
                    contacts += int(sum(sim.contact_counts))
                    distance += sim.best_progress
                    courses += 1
                    sim.reset(args.seed+courses-1)
                drive(sim,agents)
                if recorder: recorder.append(sim.snapshot())
            print(json.dumps(dict(steps=args.steps,courses=courses,distance_m=distance+sim.best_progress,
                                  contacts=contacts+int(sum(sim.contact_counts)),status=sim.status.tolist(),
                                  retained_points=len(sim.road.center),weather=sim.weather_label,
                                  simulation_speed=args.steps*.05/(time.perf_counter()-started))))
            return
        import pygame
        from procedural_ui import RaceRenderer
        from procedural_window import RaceWindow
        pygame.init()
        window = RaceWindow(args.window_size,args.fullscreen)
        screen = window.screen
        pygame.display.set_caption('CrashLearn | Grand Prix procedural | Meteo, IA & Replay')
        renderer = RaceRenderer(screen)
        clock = pygame.time.Clock()
        frame = replay.frame(0) if replay else sim.snapshot()
        ui = dict(follow=0,camera_yaw=0.,overview=args.overview,infos=True,lidar=False,trails=True,paused=False,speed=args.speed,
                  actual_speed=0.,fps=0.,replay=bool(replay),recording=bool(recorder),message='')
        accumulator = 0.
        restart_timer = 0.
        measured_time = 0.
        measured_steps = 0
        frames = 0
        index = 0
        last_recording = args.record
        running = True
        while running:
            elapsed = min(clock.tick(60)/1000.,.15)
            commands = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type in (pygame.VIDEORESIZE,pygame.WINDOWSIZECHANGED):
                    screen = window.resized()
                    renderer.screen = screen
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    commands.extend(name for name,rect in renderer.buttons.items() if rect.collidepoint(event.pos))
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_F11 or (key in (pygame.K_RETURN,pygame.K_KP_ENTER)
                                               and event.mod & pygame.KMOD_ALT):
                        if not getattr(event,'repeat',False): commands.append('fullscreen')
                        continue
                    if key == pygame.K_ESCAPE and window.fullscreen:
                        commands.append('fullscreen')
                        continue
                    mapping = {pygame.K_SPACE:'pause',pygame.K_TAB:'overview',pygame.K_l:'lidar',
                               pygame.K_t:'trails',pygame.K_h:'infos',pygame.K_c:'follow',pygame.K_w:'weather',pygame.K_p:'profile',
                               pygame.K_e:'record',pygame.K_v:'replay',pygame.K_r:'reset',pygame.K_ESCAPE:'quit',
                               pygame.K_PLUS:'faster',pygame.K_EQUALS:'faster',pygame.K_KP_PLUS:'faster',
                               pygame.K_UP:'faster',pygame.K_MINUS:'slower',pygame.K_KP_MINUS:'slower',
                               pygame.K_DOWN:'slower'}
                    if key in mapping: commands.append(mapping[key])
                    elif pygame.K_1 <= key <= pygame.K_9: ui['speed'] = key-pygame.K_0
                    elif key == pygame.K_0: ui['speed'] = 10
                    elif replay and key in (pygame.K_LEFT,pygame.K_RIGHT,pygame.K_HOME,pygame.K_END):
                        delta = 200 if event.mod & pygame.KMOD_SHIFT else 20
                        index = (0 if key == pygame.K_HOME else replay.count-1 if key == pygame.K_END else
                                 max(0,min(replay.count-1,index+(delta if key == pygame.K_RIGHT else -delta))))
                        frame = replay.frame(index)
                        accumulator = 0.
            for command in commands:
                if command == 'quit': running = False
                elif command == 'fullscreen':
                    screen = window.toggle_fullscreen()
                    renderer.screen = screen
                elif command in ('pause','overview','lidar','trails','infos'): ui[command if command != 'pause' else 'paused'] = not ui[command if command != 'pause' else 'paused']
                elif command == 'faster': ui['speed'] = min(10,ui['speed']+1)
                elif command == 'slower': ui['speed'] = max(1,ui['speed']-1)
                elif command == 'follow': ui['follow'] = (ui['follow']+1)%len(frame['states'])
                elif command.startswith('paint_'):
                    from procedural_art import PALETTE
                    renderer.colors[ui['follow']] = PALETTE[int(command.split('_')[1])]
                elif command == 'replay':
                    if replay:
                        replay.close(); replay = None
                        if sim is None:
                            sim = ProceduralSimulation(args.seed,num_cars=args.cars,weather=args.weather,
                                                       profile=args.profile,finish_distance=args.finish_distance)
                            agents = make_agents(args)
                        frame = sim.snapshot()
                    else:
                        if recorder:
                            last_recording = recorder.path
                            recorder.close(); recorder = None
                        if last_recording is None:
                            files = list(Path('recordings').glob('*.sqlite'))
                            last_recording = max(files,key=lambda p:p.stat().st_mtime) if files else None
                        if last_recording:
                            replay = RaceReplay(last_recording)
                            index = 0
                            frame = replay.frame(index)
                        else:
                            ui['message'] = 'Aucun enregistrement : appuyer sur E'
                    ui['replay'] = bool(replay)
                    ui['follow'] = 0
                    accumulator = 0.
                    renderer.last_key = None
                elif not replay:
                    if command == 'weather':
                        sim.set_weather(WEATHER_MODES[(WEATHER_MODES.index(sim.weather_mode)+1)%len(WEATHER_MODES)])
                        sim._refresh_noise()
                    elif command in ('reset','profile'):
                        if command == 'profile': sim.profile = ROAD_PROFILES[(ROAD_PROFILES.index(sim.profile)+1)%len(ROAD_PROFILES)]
                        args.seed += 1
                        sim.reset(args.seed)
                        accumulator = restart_timer = 0.
                        if recorder: recorder.append(sim.snapshot())
                    elif command == 'record':
                        if recorder:
                            name = recorder.path.name
                            last_recording = recorder.path
                            recorder.close(); recorder = None
                            ui['message'] = 'Sauvegarde : '+name
                        else:
                            path = Path('recordings')/('race-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.sqlite')
                            recorder = RaceRecorder(path)
                            last_recording = path
                            recorder.append(sim.snapshot())
                            ui['message'] = path.name
            if not ui['paused']:
                if not replay and (sim.terminated or sim.truncated):
                    restart_timer += elapsed
                    if restart_timer >= 1.5:
                        args.seed += 1
                        sim.reset(args.seed)
                        accumulator = restart_timer = 0.
                        if recorder: recorder.append(sim.snapshot())
                else:
                    accumulator = min(accumulator+elapsed*ui['speed'],.6)
                    for _ in range(12):
                        if accumulator < .05: break
                        if replay:
                            index = (index+1)%replay.count
                            frame = replay.frame(index)
                        else:
                            drive(sim,agents)
                            if recorder: recorder.append(sim.snapshot())
                        accumulator -= .05
                        measured_steps += 1
                        if not replay and (sim.terminated or sim.truncated): break
            if not replay:
                frame = sim.snapshot()
                if frame['status'][ui['follow']] != 1 and not sim.terminated:
                    ui['follow'] = next(i for i in frame['order'] if frame['status'][i] == 1)
            measured_time += elapsed
            if measured_time >= .5:
                ui['actual_speed'] = measured_steps*.05/measured_time
                measured_steps = 0; measured_time = 0.
            desired_yaw = frame['states'][ui['follow']][4]
            delta = (desired_yaw-ui['camera_yaw']+np.pi)%(2*np.pi)-np.pi
            ui['camera_yaw'] += delta*min(1.,elapsed*4*ui['speed'])
            ui['fps'] = clock.get_fps()
            ui['recording'] = bool(recorder)
            renderer.draw(frame,ui)
            pygame.display.flip()
            frames += 1
            if args.screenshot and frames == (min(180,args.frames) if args.frames else 180):
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                pygame.image.save(screen,str(args.screenshot))
            if args.frames and frames >= args.frames: running = False
        pygame.quit()
    finally:
        if recorder: recorder.close()
        if replay: replay.close()

if __name__ == '__main__':
    main()
