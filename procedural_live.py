"""Independent fixed-step simulation process for the interactive renderer.

Only display snapshots may be dropped. Every physics tick, agent decision and
recorded frame is still evaluated in order in the worker process.
"""
import multiprocessing as mp
from queue import Empty, Full
import time
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np


def interpolate_frame(previous,current,fraction):
    """Interpolate only display poses; never mutate physics/recording snapshots."""
    if (previous is None or previous['seed'] != current['seed'] or
            previous['step'] >= current['step'] or previous['status'] != current['status']):
        return current
    a,b = np.asarray(previous['states']),np.asarray(current['states'])
    t = float(np.clip(fraction,0.,1.))
    states = a+(b-a)*t
    states[:,4] = a[:,4]+((b[:,4]-a[:,4]+np.pi)%(2*np.pi)-np.pi)*t
    return dict(current,states=states)


def _run(args,commands,frames):
    recorder = None
    try:
        from procedural_demo import make_agents, drive
        from procedural_simulation import ProceduralSimulation, WEATHER_MODES, ROAD_PROFILES
        from procedural_recording import RaceRecorder
        sim = ProceduralSimulation(args.seed,num_cars=args.cars,weather=args.weather,
                                   profile=args.profile,finish_distance=args.finish_distance)
        agents = make_agents(args)
        last_recording = args.record
        if args.record:
            recorder = RaceRecorder(args.record)
            recorder.append(sim.snapshot())
        paused = True
        speed = args.speed
        ticks = 0
        ai_calls = 0
        step_seconds = 0.
        accumulator = 0.
        restart = 0.
        message = ''
        previous = time.perf_counter()
        next_publish = previous
        def publish():
            packet = dict(frame=sim.snapshot(),stamp=time.perf_counter(),ticks=ticks,ai_calls=ai_calls,step_seconds=step_seconds,recording=bool(recorder),
                          last_recording=str(last_recording) if last_recording else None,message=message)
            # A bounded latest-frame queue prevents UI stalls from growing memory.
            try: frames.put_nowait(packet)
            except Full:
                try: frames.get_nowait()
                except Empty: pass
                try: frames.put_nowait(packet)
                except Full: pass
        publish()
        while True:
            dirty = False
            while commands.poll():
                command,value = commands.recv()
                dirty = True
                if command == 'close': return
                if command == 'pace':
                    speed,paused = value
                    accumulator = 0.
                    previous = time.perf_counter()
                elif command == 'weather':
                    sim.set_weather(WEATHER_MODES[(WEATHER_MODES.index(sim.weather_mode)+1)%len(WEATHER_MODES)])
                    sim._refresh_noise()
                elif command in ('reset','profile'):
                    if command == 'profile': sim.profile = ROAD_PROFILES[(ROAD_PROFILES.index(sim.profile)+1)%len(ROAD_PROFILES)]
                    sim.reset(sim.seed+1)
                    accumulator = restart = 0.
                    if recorder: recorder.append(sim.snapshot())
                elif command in ('record','stop_record'):
                    if recorder:
                        last_recording = recorder.path
                        recorder.close()
                        recorder = None
                        message = 'Sauvegarde : '+last_recording.name
                    elif command == 'record':
                        last_recording = Path('recordings')/('race-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.sqlite')
                        recorder = RaceRecorder(last_recording)
                        recorder.append(sim.snapshot())
                        message = last_recording.name
                elif command == 'barrier':
                    # All preceding commands (notably recorder.close) are complete.
                    commands.send(dict(last_recording=str(last_recording) if last_recording else None))
            now = time.perf_counter()
            elapsed,previous = now-previous,now
            if not paused:
                if sim.terminated or sim.truncated:
                    restart += elapsed
                    if restart >= 1.5:
                        sim.reset(sim.seed+1)
                        accumulator = restart = 0.
                        if recorder: recorder.append(sim.snapshot())
                else:
                    accumulator = min(accumulator+elapsed*speed,.6)
                    # Bound each batch so user input is serviced promptly.
                    for _ in range(12):
                        if accumulator < sim.dt: break
                        ai_calls += int((sim.status == 1).sum())
                        step_started = time.perf_counter()
                        drive(sim,agents)
                        step_seconds += time.perf_counter()-step_started
                        ticks += 1
                        accumulator -= sim.dt
                        if recorder: recorder.append(sim.snapshot())
                        if sim.terminated or sim.truncated: break
            now = time.perf_counter()
            if dirty or now >= next_publish:
                publish()
                next_publish = now+1/120
            delay = min(next_publish-now,.002 if not paused else .01)
            if delay>0: time.sleep(delay)
    except BaseException:
        commands.send(dict(error=traceback.format_exc()))
    finally:
        if recorder: recorder.close()
        frames.cancel_join_thread()
        commands.close()


class LiveRace:
    def __init__(self,args):
        ctx = mp.get_context('spawn')
        self.commands,child = ctx.Pipe()
        self.frames = ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=_run,args=(args,child,self.frames),daemon=True)
        self.process.start()
        child.close()
        self.packet = None
        self.previous = None
        self.pace = None
        # Warmup includes Numba compilation / ONNX initialization, outside FPS timing.
        deadline = time.perf_counter()+120
        try:
            while self.packet is None:
                self.poll()
                if time.perf_counter()>deadline:
                    raise TimeoutError('Simulation initialization exceeded 120 seconds')
                time.sleep(.005)
        except BaseException:
            self.close()
            raise

    def poll(self):
        if self.commands.poll():
            message = self.commands.recv()
            if 'error' in message: raise RuntimeError(message['error'])
        if not self.process.is_alive():
            raise RuntimeError(f'Simulation worker stopped ({self.process.exitcode})')
        while True:
            try:
                packet = self.frames.get_nowait()
                if self.packet is None or (packet['frame']['seed'],packet['frame']['step']) != (self.packet['frame']['seed'],self.packet['frame']['step']):
                    self.previous = self.packet
                else:
                    # Keep the timestamp of the last actual physics update.
                    packet['stamp'] = self.packet['stamp']
                self.packet = packet
            except Empty: break
        return self.packet

    def visual_frame(self):
        if self.previous is None or (self.pace and self.pace[1]): return self.packet['frame']
        interval = max(.001,self.packet['stamp']-self.previous['stamp'])
        return interpolate_frame(self.previous['frame'],self.packet['frame'],
                                 (time.perf_counter()-self.packet['stamp'])/interval)

    def set_pace(self,speed,paused):
        pace = (speed,paused)
        if pace != self.pace:
            self.send('pace',pace)
            self.pace = pace

    def send(self,command,value=None):
        self.commands.send((command,value))

    def stop_recording(self):
        self.send('stop_record')
        self.send('barrier')
        if not self.commands.poll(10): raise TimeoutError('Recorder did not close')
        result = self.commands.recv()
        if 'error' in result: raise RuntimeError(result['error'])
        return Path(result['last_recording']) if result['last_recording'] else None

    def close(self):
        if self.process.is_alive():
            self.send('close')
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        self.commands.close()
        self.frames.close()
