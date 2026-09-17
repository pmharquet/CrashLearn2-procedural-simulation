"""Competition configuration and closed circuits using the existing physics."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from procedural_simulation import ProceduralSimulation, project_center
from procedural_road import EndlessRoad

ROOT = Path(__file__).resolve().parent


def tracks():
    """Deduplicate the two Mexico City directories by the actual CSV name."""
    return sorted({p.name: p for p in (ROOT / 'maps').glob('*/*_centerline.csv')}.values(),
                  key=lambda p: p.stem.lower())


@dataclass
class Driver:
    name: str
    color: tuple
    model: str = ''


@dataclass
class Competition:
    track: str = ''
    limit: int | None = 3
    seed: int = 42
    weather: str = 'cycle'
    profile: str = 'mixed'
    drivers: list = field(default_factory=list)

    def validate(self):
        if not 1 <= len(self.drivers) <= 4:
            raise ValueError('Choisissez entre 1 et 4 véhicules.')
        if self.limit is not None and (not isinstance(self.limit, int) or self.limit < 1):
            raise ValueError('Le nombre de tours / secteurs doit être positif.')
        if self.track and not Path(self.track).is_file():
            raise ValueError('Le circuit sélectionné est introuvable.')
        for driver in self.drivers:
            if not driver.name.strip():
                raise ValueError('Chaque véhicule doit avoir un nom.')
            agent_file(driver.model)


def agent_file(selection):
    path = Path(selection).expanduser().resolve()
    if not selection or not path.exists():
        raise ValueError('Sélectionnez le pilote de chaque véhicule (agent.py ou model.onnx).')
    if path.is_dir():
        path = path / 'agent.py'
    elif path.suffix.lower() == '.zip':
        return path
    elif path.suffix.lower() == '.onnx':
        if path.name != 'model.onnx':
            raise ValueError('Ce format utilise model.onnx avec son agent.py associé.')
        path = path.with_name('agent.py')
    if path.suffix.lower() != '.py' or not path.is_file():
        raise ValueError('Le modèle ONNX doit être accompagné de son agent.py (prétraitement et actions).')
    return path


class CircuitRoad:
    """Closed original centerline, uniformly scaled to the current RC road width.

    Rendering consumes the same center/width schema and unchanged road shaders.
    Stations stay periodic; lap unwrapping belongs to each vehicle, not the road.
    """
    width = EndlessRoad.width
    spacing = EndlessRoad.spacing

    def __init__(self, path):
        raw = np.loadtxt(path, delimiter=',', comments='#')
        xy = raw[:, :2]
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
        if len(xy) < 4 or not np.isfinite(raw).all():
            raise ValueError('Centre de circuit invalide.')
        # Preserve each circuit's proportions, including its road-to-turn ratio.
        source_width = float(np.median(raw[:, 2:4].sum(axis=1)))
        if source_width <= 0:
            raise ValueError('Largeur de circuit invalide.')
        xy = (xy - xy[0]) * self.width / source_width
        if np.linalg.norm(xy[-1] - xy[0]) > 1e-8:
            xy = np.vstack((xy, xy[0]))
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        self.length = float(s[-1])
        count = max(8, int(np.ceil(self.length / self.spacing)))
        stations = np.linspace(0, self.length, count, endpoint=False)
        points = np.column_stack([np.interp(stations, s, xy[:, j]) for j in (0, 1)])
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        closing_yaw = yaw[-1] + ((yaw[0]-yaw[-1]+np.pi) % (2*np.pi)-np.pi)
        self.center = np.column_stack((np.vstack((points, points[0])),
                                       np.r_[yaw, closing_yaw], np.r_[stations, self.length]))
        normal = np.column_stack((-np.sin(self.center[:, 2]), np.cos(self.center[:, 2])))
        self.left = self.center[:, :2] + normal * self.width / 2
        self.right = self.center[:, :2] - normal * self.width / 2
        self.generated = len(self.center)
        self.removed = self.rejected = self.repairs = 0
        self.sections = []

    def pose(self, station):
        station %= self.length
        return np.array([np.interp(station, self.center[:, 3], self.center[:, j]) for j in range(3)])

    def project(self, pos):
        return project_center(np.asarray(pos), self.center)

    def update(self, progress, tail_progress=None):
        pass


class CompetitionSimulation(ProceduralSimulation):
    def __init__(self, config):
        self.config = config
        self.circuit = CircuitRoad(config.track) if config.track else None
        length = self.circuit.length if self.circuit else self.sector_length
        super().__init__(config.seed, num_cars=len(config.drivers), weather=config.weather,
                         profile=config.profile, finish_distance=length*config.limit if config.limit else None)

    def reset(self, seed=0):
        super().reset(seed)
        # Leave comfortable room between RC bodies on the competition grid.
        self.starts = np.array([-2.0*(i//2) for i in range(self.num_cars)])
        self.states[:,0] = self.starts
        if self.num_cars > 1:
            self.states[:,1] = [(-.85 if i%2 == 0 else .85) for i in range(self.num_cars)]
        if self.circuit:
            self.road = self.circuit
            self.sector_length = self.circuit.length
            self.max_gap = np.inf  # Closed circuits retain all geometry, even for lapped cars.
            self.starts[:] = 0.
            for i in range(self.num_cars):
                station = -2. - 2.0*(i//2)
                x, y, yaw = self.circuit.pose(station)
                lateral = (-.85 if i % 2 == 0 else .85) if self.num_cars > 1 else 0.
                self.states[i, :2] = [x - np.sin(yaw)*lateral, y + np.cos(yaw)*lateral]
                self.states[i, 4] = yaw
                self.distances[i] = station
        self._scan_cache.clear()
        return self.observation(), self.info()

    def project_progress(self, i):
        absolute, _ = self.road.project(self.states[i, :2])
        if self.circuit:
            previous = self.distances[i]
            length = self.circuit.length
            return previous + (absolute-previous+length/2) % length-length/2
        return absolute-self.starts[i]

    def snapshot(self):
        frame = super().snapshot()
        names = [d.name for d in self.config.drivers]
        events = []
        for event in frame['events']:
            text = event['text']
            for i, name in enumerate(names):
                text = text.replace(f'Voiture {i+1} :', name + ' :')
            if self.circuit:
                text = text.replace('secteur ', 'tour ')
            events.append(dict(event, text=text))
        return dict(frame, names=names, colors=[d.color for d in self.config.drivers],
                    events=events, competition=True, unit='TOUR' if self.circuit else 'SECTEUR',
                    track_name=Path(self.config.track).stem.replace('_centerline', '') if self.circuit else 'Procédural',
                    limit=self.config.limit,
                    finish_times=[float(t) if np.isfinite(t) else None for t in self.finish_times])
