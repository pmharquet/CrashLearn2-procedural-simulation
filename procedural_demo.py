"""Live procedural race, streaming recordings and interactive replay."""
import argparse
import json
import os
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
    parser.add_argument('--renderer',choices=('auto','gpu','software'),default='auto')
    parser.add_argument('--fps',type=int,default=120,help='Frame limit; 0 = uncapped')
    parser.add_argument('--benchmark',type=Path,help='Write frame timings after 120 warmup frames')
    parser.add_argument('--fullscreen',action='store_true')
    parser.add_argument('--display',type=int,default=0,help='Monitor index, starting at 0')
    parser.add_argument('--overview',action='store_true')
    parser.add_argument('--window-size',nargs=2,type=int,metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--frames',type=int,default=0,help='Close after N rendered frames (QA)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--record',type=Path)
    group.add_argument('--replay',type=Path)
    args = parser.parse_args()
    if args.display < 0: parser.error("--display must be non-negative")
    if args.fps < 0: parser.error("--fps must be non-negative")
    return args


def main():
    args = parse_args()
    replay = RaceReplay(args.replay) if args.replay else None
    recorder = None
    sim = None
    agents = None
    live = None
    try:
        if not replay and args.headless:
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
        os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS','permonitorv2')
        import pygame
        from procedural_ui import RaceRenderer
        from procedural_window import RaceWindow
        from procedural_live import LiveRace
        pygame.init()
        window = RaceWindow(args.window_size,args.fullscreen,args.renderer,args.display)
        screen = window.screen
        pygame.display.set_caption('CrashLearn | Grand Prix procedural | Meteo, IA & Replay')
        renderer = RaceRenderer(screen,window.gpu)
        print("Renderer:",window.gpu.name if window.gpu else "Pygame software")
        clock = pygame.time.Clock()
        if not replay: live = LiveRace(args)
        frame = replay.frame(0) if replay else live.packet["frame"]
        ui = dict(follow=0,camera_yaw=0.,overview=args.overview,infos=True,lidar=False,trails=True,paused=False,speed=args.speed,
                  actual_speed=0.,fps=0.,replay=bool(replay),recording=bool(recorder),message='')
        accumulator = 0.
        measured_time = 0.
        measured_ai = 0
        measured_steps = 0
        frames = 0
        index = 0
        last_recording = args.record
        last_live_ticks = live.packet["ticks"] if live else 0
        last_ai_calls = live.packet["ai_calls"] if live else 0
        running = True
        timings = []
        measured_sizes = set()
        previous_tick = time.perf_counter()
        deadline = previous_tick
        while running:
            started = time.perf_counter()
            wall_elapsed = started-previous_tick
            elapsed = min(wall_elapsed,.15)
            previous_tick = started
            clock.tick(0)
            simulation_ticks = 0
            inference_calls = 0
            commands = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type in (pygame.VIDEORESIZE,pygame.WINDOWSIZECHANGED):
                    screen = window.resized()
                    renderer.screen = screen
                    renderer.gpu = window.gpu
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
                    renderer.gpu = window.gpu
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
                        if live is None: live = LiveRace(args)
                        frame = live.poll()['frame']
                        last_live_ticks = live.packet['ticks']
                        last_ai_calls = live.packet['ai_calls']
                    else:
                        live.set_pace(ui['speed'],True)
                        last_recording = live.stop_recording() or last_recording
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
                    measured_ai = measured_steps = 0
                    measured_time = 0.
                    ui['ai_per_second'] = ui['ticks_per_second'] = 0.
                    accumulator = 0.
                    renderer.last_key = None
                elif not replay and command in ('weather','reset','profile','record'):
                    live.send(command)
            if not replay:
                live.set_pace(ui['speed'],ui['paused'])
                packet = live.poll()
                frame = live.visual_frame()
                simulation_ticks = packet['ticks']-last_live_ticks
                last_live_ticks = packet['ticks']
                inference_calls = packet['ai_calls']-last_ai_calls
                measured_ai += inference_calls
                last_ai_calls = packet['ai_calls']
                measured_steps += simulation_ticks
                ui['recording'] = packet['recording']
                if packet['message']: ui['message'] = packet['message']
                if packet['last_recording']: last_recording = Path(packet['last_recording'])
                if frame['status'][ui['follow']] != 1 and not frame['terminated']:
                    ui['follow'] = next(i for i in frame['order'] if frame['status'][i] == 1)
            elif not ui['paused']:
                accumulator = min(accumulator+elapsed*ui['speed'],.6)
                while accumulator >= .05:
                    index = (index+1)%replay.count
                    frame = replay.frame(index)
                    accumulator -= .05
                    measured_steps += 1
                    simulation_ticks += 1
            measured_time += wall_elapsed
            if measured_time >= .5:
                ui['actual_speed'] = measured_steps*.05/measured_time
                ui['ticks_per_second'] = measured_steps/measured_time
                ui['ai_per_second'] = measured_ai/measured_time
                measured_ai = 0
                measured_steps = 0; measured_time = 0.
            desired_yaw = frame['states'][ui['follow']][4]
            delta = (desired_yaw-ui['camera_yaw']+np.pi)%(2*np.pi)-np.pi
            ui['camera_yaw'] += delta*min(1.,elapsed*4*ui['speed'])
            ui['fps'] = clock.get_fps()
            if replay: ui['recording'] = False
            render_start = time.perf_counter()
            renderer.draw(frame,ui)
            if args.screenshot and frames+1 == (min(180,args.frames) if args.frames else 180):
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                if window.gpu: window.gpu.screenshot(args.screenshot)
                else: pygame.image.save(screen,str(args.screenshot))
            present_start = time.perf_counter()
            pygame.display.flip()
            finished = time.perf_counter()
            if args.benchmark and frames >= 120:
                measured_sizes.add(screen.get_size())
                timings.append((wall_elapsed,render_start-started,present_start-render_start,
                                finished-present_start,simulation_ticks*.05,inference_calls))
            frames += 1
            if args.frames and frames >= args.frames: running = False
            if args.fps:
                deadline += 1/args.fps
                if finished-deadline > .25: deadline = finished
                delay = deadline-time.perf_counter()
                if delay>0: time.sleep(delay)
        if args.benchmark:
            if not timings:
                raise ValueError('Benchmark requires --frames > 120 (warmup)')
            values = np.asarray(timings)
            report = dict(renderer=window.gpu.name if window.gpu else 'software',
                          resolution=list(screen.get_size()),resolutions=sorted(measured_sizes),
                          speed=args.speed,frames=len(values),
                          fps=len(values)/values[:,0].sum(),actual_speed=values[:,4].sum()/values[:,0].sum(),
                          frame_ms_p50=float(np.percentile(values[:,0]*1000,50)),
                          frame_ms_p95=float(np.percentile(values[:,0]*1000,95)),
                          ticks_per_second=values[:,4].sum()/.05/values[:,0].sum(),
                          ai_calls_per_second=values[:,5].sum()/values[:,0].sum(),
                          worker_step_ms=live.packet['step_seconds']/max(1,live.packet['ticks'])*1000 if live else None,
                          ui_update_ms=float(values[:,1].mean()*1000),
                          render_ms=float(values[:,2].mean()*1000),
                          present_ms=float(values[:,3].mean()*1000))
            args.benchmark.parent.mkdir(parents=True,exist_ok=True)
            args.benchmark.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2))
        if window.gpu: window.gpu.release()
        pygame.quit()
    finally:
        if recorder: recorder.close()
        if replay: replay.close()
        if live: live.close()

if __name__ == '__main__':
    main()
