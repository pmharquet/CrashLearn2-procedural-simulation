"""Gymnasium training on the shared procedural race engine, including opponents."""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from procedural_simulation import ProceduralSimulation, load_agent, WEATHER_MODES, ROAD_PROFILES

class ProceduralEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self,max_steps=4000,num_cars=1,weather='cycle',profile='mixed',agent_path=None,
                 finish_distance=None):
        self.sim = ProceduralSimulation(max_steps=max_steps,num_cars=num_cars,weather=weather,
                                        profile=profile,finish_distance=finish_distance)
        self.opponents = [load_agent(agent_path) for _ in range(num_cars-1)]
        self.action_space = gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.,1.,(131,),dtype=np.float32)

    @staticmethod
    def encode(obs, legacy=False):
        base = np.concatenate((obs['lidar']/15.,[np.clip(obs['velocity']/10.,-1,1),
                                               np.clip(obs['steering']/.4189,-1,1)]))
        if legacy:
            return base.astype(np.float32)
        values = []
        for i in range(4):
            opp = obs['opponents'][i]
            values.extend((np.clip(opp['rel_x']/100.,-1,1),np.clip(opp['rel_y']/100.,-1,1),
                           np.sin(opp['rel_yaw']),np.cos(opp['rel_yaw']),
                           np.clip(opp['velocity']/10.,-1,1),
                           np.clip(opp['rel_dist']/100.,0,1),float(opp['active'])))
        return np.concatenate((base,values,[obs['friction']])).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        obs,info = self.sim.reset(int(self.np_random.integers(0,2**31)))
        return self.encode(obs),info

    def step(self,action):
        action = np.asarray(action,dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected two finite normalized actions')
        action = np.clip(action,-1,1)
        actions = [(4.+6.*action[0],.4189*action[1])]
        for i,agent in enumerate(self.opponents,1):
            actions.append(agent.predict(self.sim.observation(i),self.sim.info())
                           if self.sim.status[i] == 1 else (0.,0.))
        obs,reward,terminated,truncated,info = self.sim.step(actions)
        # Training concerns car 0; the demonstration can continue following others.
        terminated = terminated or self.sim.status[0] != 1
        return self.encode(obs),reward,bool(terminated),truncated,info

if __name__ == '__main__':
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=200000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cars',type=int,choices=range(1,5),default=4)
    parser.add_argument('--weather',choices=WEATHER_MODES,default='cycle')
    parser.add_argument('--profile',choices=ROAD_PROFILES,default='mixed')
    parser.add_argument('--agent',type=Path)
    parser.add_argument('--finish-distance',type=float)
    parser.add_argument('--output',type=Path,default=Path('recordings/procedural_ppo'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    env = ProceduralEnv(num_cars=args.cars,weather=args.weather,profile=args.profile,
                        agent_path=args.agent,finish_distance=args.finish_distance)
    try:
        model = PPO('MlpPolicy',env,seed=args.seed,verbose=1,n_steps=1024,batch_size=64)
        model.learn(total_timesteps=args.steps)
        model.save(args.output)
    finally:
        env.close()
