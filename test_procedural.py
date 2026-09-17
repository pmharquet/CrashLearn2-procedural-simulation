"""Geometry, streaming, sensor and training regression checks."""
import unittest
import numpy as np
from procedural_simulation import EndlessRoad, ProceduralSimulation, scan_walls
from procedural_train import ProceduralEnv

class ProceduralTests(unittest.TestCase):
    def test_actual_half_turn_and_direct_triple_reversals(self):
        from procedural_road import curvature_profile, integrate_profile
        for sign in (-1,1):
            k = curvature_profile([(sign*180.,3.8)])
            points = integrate_profile(np.zeros(4),k)
            self.assertAlmostEqual(points[-1,2],sign*np.pi,places=10)
            self.assertLess(points[-1,0]-points[-2,0],-.49)
            self.assertLessEqual(np.max(abs(k)),1/3.8)
        road = EndlessRoad(0)
        k = curvature_profile(road.commands('triple chicane',1))
        nonzero = k[abs(k)>1e-10]
        self.assertEqual(np.count_nonzero(np.diff(np.sign(nonzero))),4)
        # Each reversal crosses zero in <=1 m, without a straight inserted.
        transitions = np.flatnonzero(np.diff(np.sign(nonzero)))
        self.assertTrue(all(abs(nonzero[i])+abs(nonzero[i+1])>.1 for i in transitions))
        self.assertLessEqual(np.count_nonzero(abs(k)<1e-10),5)

    def test_tight_hairpin_gap_and_turn_beyond_180(self):
        from procedural_road import curvature_profile, integrate_profile
        for sign in (-1,1):
            k = curvature_profile([(sign*180.,2.55)])
            points = integrate_profile(np.zeros(4),k)
            gap = abs(points[-1,1])-EndlessRoad.width
            self.assertGreater(gap,.25)
            self.assertLess(gap,.45)
            self.assertAlmostEqual(np.max(abs(k)),1/2.55)
            self.assertAlmostEqual(points[-1,2],sign*np.pi)
        points = integrate_profile(np.zeros(4),curvature_profile([(230.,8.),(-70.,6.)]))
        self.assertGreater(np.rad2deg(points[:,2].max()),225.)
        self.assertLess(np.rad2deg(points[-1,2]),180.)

    def test_hidden_repair_preserves_every_published_rail(self):
        from unittest.mock import patch
        road = EndlessRoad(0)
        before = road.center.copy()
        append = road._append_section
        calls = 0
        def fail_once():
            nonlocal calls
            calls += 1
            return False if calls == 1 else append()
        with patch.object(road,'_append_section',side_effect=fail_once):
            road.update(150.)
        self.assertGreater(road.repairs,0)
        retained = before[before[:,3]>=road.center[0,3]]
        np.testing.assert_array_equal(retained,road.center[:len(retained)])

    def test_rails_do_not_intersect_independent_polygon_check(self):
        # Independent orientation test on the actual rails, not tube_clear.
        def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]
        for seed in range(10):
            road = EndlessRoad(seed)
            for progress in range(0,1000,200):
                road.update(progress)
                boundary = np.concatenate((road.left,road.right[::-1],road.left[:1]))
                a,b = boundary[:-1],boundary[1:]
                ab = b-a
                for first in range(0,len(a),100):
                    c,d = a[first:first+100,None,:],b[first:first+100,None,:]
                    cd = d-c
                    side1 = cross(ab,c-a)*cross(ab,d-a)
                    side2 = cross(cd,a-c)*cross(cd,b-c)
                    self.assertFalse(np.any((side1 < -1e-10)&(side2 < -1e-10)),(seed,progress))

    def test_stream_geometry_and_memory(self):
        for seed in range(20):
            road = EndlessRoad(seed)
            headings = []
            for progress in range(0,2000,100):
                road.update(progress)
                p = road.center
                self.assertLessEqual(len(p),424)
                self.assertLessEqual(len(road._plan),1200)
                np.testing.assert_allclose(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1),.5,atol=1e-9)
                self.assertLessEqual(np.max(np.abs(np.diff(p[:,2]))/.5),1/2.55+1e-9)
                np.testing.assert_allclose(np.linalg.norm(road.left-road.right,axis=1),4.8,atol=1e-10)
                self.assertGreaterEqual(p[-1,3],progress+road.ahead)
                self.assertLessEqual(p[0,3],progress-road.behind+.5)
                headings.append(p[-1,2])
            self.assertGreater(np.std(headings),.35)

    def test_mixture_and_lidar_contract(self):
        from procedural_simulation import LIDAR_ANGLES
        road = EndlessRoad(42)
        curvature = []
        for progress in range(0,10000,50):
            road.update(progress)
            curvature.extend(np.abs(np.diff(road.center[:,2]))/.5)
        self.assertTrue(set(EndlessRoad.names).issubset(road.motif_counts))
        self.assertGreater(np.mean(np.asarray(curvature)<.01), .15)
        self.assertGreater(np.max(curvature), .22)
        np.testing.assert_array_equal(LIDAR_ANGLES,np.linspace(-np.pi,np.pi,100))
        # A displaced car must see asymmetric distances to the actual rails.
        sim = ProceduralSimulation(42)
        sim.state[1] = .8
        scan = sim.observation()['lidar']
        self.assertAlmostEqual(float(scan[74]),1.6/abs(np.sin(LIDAR_ANGLES[74])),places=4)
        self.assertAlmostEqual(float(scan[25]),3.2/abs(np.sin(LIDAR_ANGLES[25])),places=4)

    def test_seed_and_seam(self):
        a,b = EndlessRoad(12),EndlessRoad(12)
        np.testing.assert_array_equal(a.center,b.center)
        old = a.center.copy()
        a.update(20)
        retained = old[old[:,3]>=a.center[0,3]]
        np.testing.assert_array_equal(retained,a.center[:len(retained)])
        self.assertFalse(np.array_equal(b.center,EndlessRoad(13).center))

    def test_lidar_straight(self):
        sim = ProceduralSimulation(1)
        scan = sim.observation()['lidar']
        self.assertEqual(scan.shape,(100,))
        self.assertEqual(scan.dtype,np.float32)
        angles = np.linspace(-np.pi,np.pi,100)
        for i in (24,25,74,75):
            self.assertAlmostEqual(float(scan[i]),2.4/abs(np.sin(angles[i])),places=4)

    def test_crash_and_invalid_actions(self):
        sim = ProceduralSimulation()
        with self.assertRaises(ValueError): sim.step([np.nan,0])
        sim.state[4] = np.pi/2
        for _ in range(200):
            _,_,end,_,_ = sim.step([4,0])
            if end: break
        self.assertTrue(sim.collision)
        with self.assertRaises(RuntimeError): sim.step([1,0])

    def test_training_timeout_and_seed(self):
        env = ProceduralEnv(max_steps=2)
        first,_ = env.reset(seed=7)
        again,_ = env.reset(seed=7)
        np.testing.assert_array_equal(first,again)
        self.assertTrue(env.observation_space.contains(first))
        self.assertFalse(env.step([0,0])[3])
        self.assertTrue(env.step([0,0])[3])

class RaceSystemsTests(unittest.TestCase):
    def test_weather_seed_noise_and_read_order(self):
        a = ProceduralSimulation(seed=9,weather='stress')
        b = ProceduralSimulation(seed=9,weather='stress')
        for _ in range(20):
            a.observation(); a.observation(); a.snapshot()
            oa = a.step([3.,.08])[0]
            ob = b.step([3.,.08])[0]
            np.testing.assert_array_equal(oa['lidar'],ob['lidar'])
            np.testing.assert_array_equal(a.state,b.state)
        self.assertGreaterEqual(a.friction,.55)
        self.assertLessEqual(a.friction,1.)
        np.testing.assert_array_equal(a.observation()['lidar'],a.observation()['lidar'])
        self.assertGreater(a.noise_level,.001)
        a.steps = 399
        a.set_weather('cycle')
        a.step([3.,0.])
        self.assertEqual(a.target_friction,.78)
        self.assertNotEqual(a.friction,a.target_friction)  # gradual transition

    def test_friction_changes_physics(self):
        from procedural_simulation import advance
        dry = np.array([0.,0.,.12,4.,0.,0.,0.])
        wet = dry.copy()
        for _ in range(100):
            dry = advance(dry,4.,.12,1.)
            wet = advance(wet,4.,.12,.55)
        self.assertGreater(np.linalg.norm(dry-wet),.001)

    def test_opponents_wall_lidar_and_self_mask(self):
        sim = ProceduralSimulation(num_cars=4)
        obs = sim.observation(0)
        self.assertEqual(obs['opponents'][0]['active'],0)
        self.assertEqual(sum(v['active'] for v in obs['opponents'].values()),3)
        self.assertAlmostEqual(obs['opponents'][1]['rel_y'],1.1)
        scan = obs['lidar'].copy()
        sim.states[1,0] += 3.
        np.testing.assert_array_equal(scan,sim.observation(0)['lidar'])
        sim._retire(1,'test')
        self.assertEqual(sim.observation(0)['opponents'][1]['active'],0)

    def test_rectangle_contact_and_impulse(self):
        from procedural_simulation import contact_vector
        sim = ProceduralSimulation(num_cars=2)
        sim.states[:,:2] = [[0.,0.],[.5,0.]]
        sim.states[:,3] = [4.,0.]
        self.assertGreater(np.linalg.norm(contact_vector(*sim.states)),0.)
        sim._resolve_contacts()
        self.assertTrue(all(sim.vehicle_hits))
        self.assertLess(sim.states[0,3],4.)
        self.assertGreater(sim.states[1,3],0.)
        self.assertAlmostEqual(np.linalg.norm(contact_vector(*sim.states)),0.)

    def test_wall_recovery_and_dnf(self):
        sim = ProceduralSimulation()
        sim.state[4] = np.pi/2
        for _ in range(70):
            sim.step([4.,0.])
            if sim.wall_hits[0]: break
        self.assertTrue(sim.wall_hits[0])
        self.assertFalse(sim.terminated)
        before = sim.state[1]
        for _ in range(10): sim.step([-2.,0.])
        self.assertLess(sim.state[1],before)
        idle = ProceduralSimulation()
        for _ in range(80): idle.step([0.,0.])
        self.assertTrue(idle.terminated)
        self.assertEqual(idle.reasons[0],'immobile 4 s')

    def test_finish_ranks_and_sector(self):
        sim = ProceduralSimulation(num_cars=2,finish_distance=.1)
        for _ in range(30):
            sim.step([[3.,0.],[0.,0.]])
            if sim.status[0] == 2: break
        self.assertEqual(sim.status[0],2)
        self.assertEqual(sim.ranks()[0],1)
        self.assertFalse(sim.terminated)
        self.assertGreater(sim.finish_times[0],0.)
        sector = ProceduralSimulation()
        point = sector.road.center[np.argmin(abs(sector.road.center[:,3]-99.5))]
        sector.state[:2] = point[:2]
        sector.state[4] = point[2]
        sector.state[3] = 4.
        sector.best[0] = 99.5
        for _ in range(10): sector.step([4.,0.])
        self.assertEqual(sector.sectors[0],1)
        self.assertEqual(len(sector.sector_times[0]),1)
        self.assertGreater(sector.sector_times[0][0],0.)

    def test_record_replay_exact_seek_and_no_overwrite(self):
        import tempfile
        from pathlib import Path
        from procedural_recording import RaceRecorder,RaceReplay
        sim = ProceduralSimulation(num_cars=2,weather='stress')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'race.sqlite'
            with RaceRecorder(path) as recorder:
                first = sim.snapshot(); recorder.append(first)
                for _ in range(8):
                    sim.step([[3.,0.],[2.,0.]])
                    recorder.append(sim.snapshot())
                last = sim.snapshot()
            replay = RaceReplay(path)
            try:
                self.assertEqual(replay.count,9)
                self.assertEqual(replay.frame(8),last)
                self.assertEqual(replay.frame(0),first)
                self.assertEqual(replay.frame(999),last)
            finally: replay.close()
            with self.assertRaises(FileExistsError): RaceRecorder(path)

    def test_streaming_retires_distant_car(self):
        sim = ProceduralSimulation(num_cars=2)
        point = sim.road.center[np.argmin(abs(sim.road.center[:,3]-90.))]
        sim.states[0,:2] = point[:2]
        sim.states[0,4] = point[2]
        sim.states[1,:2] = [-20.,0.]
        sim.step([[0.,0.],[0.,0.]])
        self.assertEqual(sim.status[1],0)
        self.assertEqual(sim.reasons[1],'retard > 100 m')
        self.assertLessEqual(len(sim.road.center),625)
        self.assertLess(sim.road.center[0,3],90.)
        self.assertGreaterEqual(sim.road.center[-1,3],250.)

    def test_interactive_controls_record_replay(self):
        import os
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from procedural_recording import RaceReplay
        with patch.dict(os.environ,{'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}):
            import pygame
            import procedural_demo
            keys = [pygame.K_F11,pygame.K_RETURN,pygame.K_F11,pygame.K_ESCAPE,
                    pygame.K_l,pygame.K_w,pygame.K_p,pygame.K_c,pygame.K_SPACE,
                    pygame.K_SPACE,pygame.K_e,pygame.K_v,pygame.K_RIGHT,pygame.K_HOME,
                    pygame.K_END,pygame.K_v,pygame.K_r,pygame.K_TAB,pygame.K_h,pygame.K_h,pygame.K_PLUS,
                    pygame.K_MINUS,pygame.K_ESCAPE]
            def events():
                if not keys: return []
                key = keys.pop(0)
                return [pygame.event.Event(pygame.KEYDOWN,key=key,mod=pygame.KMOD_ALT if key == pygame.K_RETURN else 0)]
            with tempfile.TemporaryDirectory() as root:
                path = Path(root)/'controls.sqlite'
                agent = Path(root)/'agent'
                agent.mkdir()
                (agent/'model.onnx').touch()
                (agent/'agent.py').write_text('class Agent:\n    def predict(self,obs,info): return 3.,0.\n')
                argv = ['procedural_demo.py','--agent',str(agent),'--cars','1','--record',str(path),'--frames','30']
                with patch.object(sys,'argv',argv),patch.object(pygame.event,'get',events):
                    procedural_demo.main()
                replay = RaceReplay(path)
                try:
                    self.assertGreaterEqual(replay.count,2)
                    self.assertEqual(replay.frame(replay.count-1)['profile'],'flowing')
                finally: replay.close()

    def test_resize_does_not_recreate_window_and_fullscreen_restores_size(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}):
            import pygame
            from procedural_window import RaceWindow
            pygame.init()
            try:
                window = RaceWindow((800,600))
                # Mimic SDL's automatic resize before delivering the notification.
                pygame.display.set_mode((720,480),pygame.RESIZABLE)
                with patch.object(pygame.display,'set_mode',wraps=pygame.display.set_mode) as change:
                    for _ in range(4): window.resized()
                    change.assert_not_called()
                self.assertEqual(window.windowed_size,(720,480))
                window.toggle_fullscreen()
                self.assertTrue(window.fullscreen)
                window.resized()
                self.assertEqual(window.windowed_size,(720,480))
                window.toggle_fullscreen()
                self.assertFalse(window.fullscreen)
                self.assertEqual(window.screen.get_size(),(720,480))
            finally: pygame.quit()

    def test_responsive_render_and_button_coordinates(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}):
            import pygame
            from procedural_ui import RaceRenderer
            pygame.init()
            try:
                frame = ProceduralSimulation(num_cars=4,weather='stress').snapshot()
                ui = dict(follow=0,camera_yaw=0.,overview=False,lidar=True,paused=False,
                          speed=3,actual_speed=3.,fps=60.,replay=False,recording=False,message='')
                for size in ((640,480),(800,600),(480,800),(1280,720),(1920,1080),(2560,1080)):
                    surface = pygame.Surface(size)
                    renderer = RaceRenderer(surface)
                    renderer.draw(frame,ui)
                    self.assertIs(renderer.screen,surface)
                    self.assertEqual(len(renderer.buttons),18)
                    for name,rect in renderer.buttons.items():
                        self.assertTrue(surface.get_rect().contains(rect),(size,name,rect))
                        self.assertEqual([n for n,r in renderer.buttons.items() if r.collidepoint(rect.center)],[name])
                    ui['overview'] = True
                    with patch.object(renderer,'text',wraps=renderer.text) as labels:
                        renderer.draw(frame,ui)
                    drawn = [call.args[0] for call in labels.call_args_list]
                    for label in ('CLASSEMENT','ADHERENCE / CAPTEURS','EVENEMENTS','POSITION 1/4'):
                        self.assertIn(label,drawn,(size,label))
                    for rect in renderer.buttons.values():
                        self.assertTrue(surface.get_rect().contains(rect))
                    ui['infos'] = False
                    with patch.object(renderer,'text',wraps=renderer.text) as labels:
                        renderer.draw(frame,ui)
                    self.assertNotIn('CLASSEMENT',[call.args[0] for call in labels.call_args_list])
                    self.assertIn('infos',renderer.buttons)
                    ui['infos'] = True
                    ui['overview'] = False
            finally: pygame.quit()

    def test_dynamic_weather_changes_regime_and_schedule(self):
        a = ProceduralSimulation(seed=7,weather='dynamic')
        b = ProceduralSimulation(seed=7,weather='dynamic')
        kinds,intervals = [],[]
        for _ in range(30):
            self.assertEqual(a.target_friction,b.target_friction)
            self.assertEqual(a.weather_next_step,b.weather_next_step)
            kinds.append(a.dynamic_kind)
            intervals.append(a.weather_next_step-a.steps)
            self.assertGreaterEqual(intervals[-1],240)
            self.assertLessEqual(intervals[-1],640)
            a.steps = a.weather_next_step-1
            b.steps = b.weather_next_step-1
            a.step([0.,0.]); b.step([0.,0.])
        self.assertEqual(set(kinds),{0,1,2})
        self.assertTrue(all(a != b for a,b in zip(kinds,kinds[1:])))
        self.assertGreater(len(set(intervals)),15)
        self.assertTrue(any(k == 2 for k in kinds))
        a.friction = .95
        a._refresh_noise()
        self.assertEqual(a.rain,0.)
        self.assertEqual(a.noise_level,.001)
        self.assertEqual(a.dropout_rate,0.)
        a.friction = .6
        a._refresh_noise()
        self.assertGreater(a.rain,.8)
        self.assertGreater(a.noise_level,.04)

    def test_wet_patches_are_fixed_when_road_streams(self):
        from procedural_ui import wet_patches
        road = EndlessRoad(42)
        before = wet_patches(road.center,road.width,42)
        road.update(20.)
        after = wet_patches(road.center,road.width,42)
        common = set(before)&set(after)
        self.assertGreater(len(common),10)
        for key in common:
            np.testing.assert_array_equal(before[key],after[key])

    def test_rain_has_independent_positions_and_velocities(self):
        from procedural_ui import RainField
        rain = RainField(42)
        starts,ends,brightness = rain.segments(3.2,(1280,900),1.)
        duplicate = RainField(42).segments(3.2,(1280,900),1.)[0]
        np.testing.assert_array_equal(starts,duplicate)
        self.assertLess(abs(np.corrcoef(rain.origins.T)[0,1]),.2)
        self.assertGreater(np.std(rain.wind),20.)
        self.assertGreater(np.std(rain.speeds),70.)
        self.assertGreater(len(set(starts[:,0].astype(int))),200)
        self.assertGreater(np.std(np.linalg.norm(ends-starts,axis=1)),3.)
        self.assertTrue(np.all(ends[:,1]>starts[:,1]))

    def test_profiles_and_input_validation(self):
        for profile in ('mixed','flowing','technical'):
            road = EndlessRoad(42,profile)
            road.update(10000)
            self.assertLess(len(road.center),425)
        for cars in (0,5):
            with self.assertRaises(ValueError): ProceduralSimulation(num_cars=cars)
        with self.assertRaises(ValueError): ProceduralSimulation(weather='invalid')
        with self.assertRaises(ValueError): ProceduralSimulation(finish_distance=-1)
        sim = ProceduralSimulation(num_cars=4)
        with self.assertRaises(ValueError): sim.step([3.,0.])


if __name__ == '__main__': unittest.main()
