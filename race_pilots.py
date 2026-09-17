"""One process per external pilot: isolated imports, state and failure handling."""
import importlib.util
import multiprocessing as mp
import os
import sys
import time
import numpy as np
from race_competition import agent_file


def _pilot(selection, channel):
    try:
        path = agent_file(selection)
        os.chdir(path.parent)
        sys.path.insert(0, str(path.parent))
        if path.suffix.lower() == '.zip':
            from stable_baselines3 import PPO
            from procedural_train import ProceduralEnv
            class PPOAgent:
                def __init__(self):
                    self.model = PPO.load(path, device='cpu')
                    size = self.model.observation_space.shape[0]
                    if size not in (102,131):
                        raise ValueError('Le PPO doit utiliser le schéma du simulateur (102 ou 131 entrées).')
                    self.legacy = size == 102
                def predict(self, obs, info):
                    action,_ = self.model.predict(ProceduralEnv.encode(obs,legacy=self.legacy),deterministic=True)
                    return float(4+6*action[0]),float(.4189*action[1])
            agent = PPOAgent()
        else:
            spec = importlib.util.spec_from_file_location('external_race_pilot', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            agent = module.Agent()
        channel.send(('ready', None))
        while True:
            request = channel.recv()
            if request is None:
                return
            action = np.asarray(agent.predict(*request), dtype=float)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError('predict doit renvoyer deux nombres finis : vitesse, direction')
            channel.send(('action', action.tolist()))
    except EOFError:
        pass
    except BaseException as error:
        try:
            channel.send(('error', f'{type(error).__name__}: {error}'))
        except (OSError, EOFError):
            pass
    finally:
        channel.close()


class PilotPool:
    def __init__(self, config, sim, stop=None):
        self.pilots = []
        self.cancel = stop
        ctx = mp.get_context('spawn')
        try:
            for driver in config.drivers:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_pilot, args=(driver.model, child), daemon=True)
                process.start()
                child.close()
                self.pilots.append((process, parent))
            deadline = time.monotonic() + 90
            for i in range(len(self.pilots)):
                self.receive(i, deadline, 'ready')
            # Validate the real observation contract before countdown.
            self.actions(sim, strict=True)
        except BaseException:
            self.close()
            raise

    def receive(self, i, deadline, expected='action'):
        process, channel = self.pilots[i]
        while not channel.poll(.03):
            if self.cancel is not None and self.cancel.is_set():
                raise InterruptedError('chargement annulé')
            if time.monotonic() >= deadline:
                raise TimeoutError('pilote sans réponse')
        kind, value = channel.recv()
        if kind != expected:
            raise ValueError(value or 'réponse du pilote invalide')
        return value

    def actions(self, sim, strict=False):
        actions = [(0., 0.) for _ in self.pilots]
        active = np.flatnonzero(sim.status == 1)
        info = sim.info()
        for i in active:
            try:
                self.pilots[i][1].send((sim.observation(i), info))
            except (OSError, EOFError):
                if strict:
                    raise ValueError(f'{sim.config.drivers[i].name} : pilote arrêté')
                sim._retire(i, 'pilote arrêté')
        deadline = time.monotonic() + (30 if strict else 2)
        for i in active:
            if sim.status[i] != 1:
                continue
            try:
                actions[i] = self.receive(i, deadline)
            except (OSError, EOFError, ValueError, TimeoutError) as error:
                if strict:
                    raise ValueError(f'{sim.config.drivers[i].name} : {error}') from error
                sim._retire(i, str(error)[:60])
                self.stop(i)
        return actions

    def stop(self, i):
        process, channel = self.pilots[i]
        if process.is_alive():
            process.terminate()
        process.join(1)
        channel.close()

    def close(self):
        for i in range(len(self.pilots)):
            self.stop(i)
