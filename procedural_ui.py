"""Pygame race renderer. Weather graphics never alter simulation randomness."""
from collections import deque
import numpy as np
import pygame
from procedural_simulation import LIDAR_ANGLES

COLORS = [(65,238,193),(255,183,89),(127,171,255),(225,135,229)]
TEXT = (217,233,235)
MUTED = (143,169,177)

class RainField:
    """Independent, reproducible particles; no shared grid or repeated columns."""
    def __init__(self,seed):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed),731]))
        self.origins = rng.random((240,2))
        self.speeds = rng.uniform(300.,650.,240)
        self.wind = rng.uniform(-155.,-55.,240)
        self.lengths = rng.uniform(7.,21.,240)
        self.brightness = rng.uniform(.65,1.,240)

    def segments(self,time,size,intensity):
        count = int(240*np.clip(intensity,0.,1.))
        width,height = size
        velocity = np.column_stack((self.wind[:count],self.speeds[:count]))
        starts = (self.origins[:count]*[width,height]+time*velocity)%[width,height]
        directions = velocity/np.linalg.norm(velocity,axis=1)[:,None]
        ends = starts+directions*self.lengths[:count,None]
        return starts,ends,self.brightness[:count]


def wet_patches(center,width,seed):
    """Small wet areas off the center marking, keyed by absolute road station.

    Streaming away old points never reindexes these details or moves them.
    """
    result = {}
    start = int(np.ceil((center[0,3]+1.)/7.))
    end = int(np.floor((center[-1,3]-1.)/7.))
    for station in range(start,end+1):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed),station%2**32,83]))
        distance = station*7.
        pos = np.array([np.interp(distance,center[:,3],center[:,0]),
                        np.interp(distance,center[:,3],center[:,1])])
        yaw = np.interp(distance,center[:,3],center[:,2])
        tangent = np.array([np.cos(yaw),np.sin(yaw)])
        normal = np.array([-np.sin(yaw),np.cos(yaw)])
        lateral = float(rng.choice([-1,1]))*rng.uniform(.75,width/2-.65)
        length,radius = rng.uniform(.45,.9),rng.uniform(.15,.27)
        angles = np.linspace(0,2*np.pi,8,endpoint=False)
        result[station] = (pos+lateral*normal + np.cos(angles)[:,None]*length*tangent
                           + np.sin(angles)[:,None]*radius*normal)
    return result


class RaceRenderer:
    def __init__(self,screen):
        self.screen = screen
        self.font = pygame.font.SysFont('consolas',17)
        self.small = pygame.font.SysFont('consolas',14)
        self.big = pygame.font.SysFont('consolas',27,bold=True)
        self.huge = pygame.font.SysFont('consolas',38,bold=True)
        self.trails = [deque(maxlen=650) for _ in range(4)]
        self.last_key = None
        self.buttons = {}
        self._canvas = None
        self._rain_seed = None
        self._rain_field = None
        self._patch_key = None
        self._patches = {}

    def text(self,text,x,y,color=TEXT,font=None):
        self.screen.blit((font or self.font).render(str(text),True,color),(int(x),int(y)))

    def panel(self,rect):
        surf = pygame.Surface((rect[2],rect[3]),pygame.SRCALPHA)
        surf.fill((8,19,28,232))
        self.screen.blit(surf,rect[:2])
        pygame.draw.rect(self.screen,(40,64,75),rect,1,border_radius=8)

    def button(self,name,label,rect,active=False):
        self.buttons[name] = pygame.Rect(rect)
        pygame.draw.rect(self.screen,(34,93,90) if active else (25,43,56),rect,border_radius=5)
        self.text(label,rect[0]+10,rect[1]+10,COLORS[0] if active else TEXT,self.small)

    def weather_icon(self,x,y,rain,t):
        if rain < .1:
            pygame.draw.circle(self.screen,(255,202,103),(x,y),12)
            for a in np.arange(8)*np.pi/4:
                pygame.draw.line(self.screen,(255,202,103),(x+17*np.cos(a),y+17*np.sin(a)),
                                 (x+24*np.cos(a),y+24*np.sin(a)),2)
        else:
            for dx,dy,r in ((-13,0,11),(0,-5,15),(15,2,11)):
                pygame.draw.circle(self.screen,(159,180,198),(x+dx,y+dy),r)
            for i in range(4):
                yy = y+15+(t*50+i*9)%22
                pygame.draw.line(self.screen,(95,170,230),(x-18+i*11,yy),(x-21+i*11,yy+7),2)

    def draw(self,frame,ui):
        # Keep the HUD entirely visible on small/short windows while preserving
        # aspect ratio. Larger windows use native pixels and anchored panels.
        display = self.screen
        width,height = display.get_size()
        factor = min(1.,width/1280.,height/900.)
        if factor >= 1.:
            self._draw_frame(frame,ui)
            return
        logical_size = (int(np.ceil(width/factor)),int(np.ceil(height/factor)))
        if self._canvas is None or self._canvas.get_size() != logical_size:
            self._canvas = pygame.Surface(logical_size)
        self.screen = self._canvas
        try:
            self._draw_frame(frame,ui)
        finally:
            self.screen = display
        pygame.transform.smoothscale(self._canvas,(width,height),display)
        sx,sy = width/logical_size[0],height/logical_size[1]
        self.buttons = {name:pygame.Rect(round(rect.x*sx),round(rect.y*sy),
                                        max(1,round(rect.width*sx)),max(1,round(rect.height*sy)))
                        for name,rect in self.buttons.items()}

    def _draw_frame(self,frame,ui):
        screen = self.screen
        width,height = screen.get_size()
        states = np.asarray(frame['states'])
        count = len(states)
        follow = min(ui['follow'],count-1)
        key = (frame['seed'],frame['step'])
        if self.last_key != key:
            if self.last_key is None or key[0] != self.last_key[0] or key[1] < self.last_key[1]:
                for trail in self.trails: trail.clear()
            for i in range(count):
                if frame['status'][i] == 1: self.trails[i].append(states[i,:2].copy())
            self.last_key = key
        pos = states[follow,:2]
        center = np.asarray(frame['center'])
        yaw = ui['camera_yaw']
        c,s = np.cos(yaw),np.sin(yaw)
        scale = min(width/70,height/63)
        if ui['overview']:
            # Reserve both information columns; fit the road between them.
            margin = 310 if ui.get('infos',True) else 32
            viewport = pygame.Rect(margin,120,width-2*margin,height-235)
            lo,hi = center[:,:2].min(axis=0)-6,center[:,:2].max(axis=0)+6
            scale = min(viewport.width/max(1.,hi[0]-lo[0]),viewport.height/max(1.,hi[1]-lo[1]))
            view_origin = (lo+hi)/2
            def project(points):
                p = np.asarray(points)-view_origin
                return np.column_stack((viewport.centerx+p[:,0]*scale,
                                        viewport.centery-p[:,1]*scale)).astype(int)
        else:
            def project(points):
                p = np.asarray(points)-pos
                return np.column_stack((width*.53+(-s*p[:,0]+c*p[:,1])*scale,
                                        height*.64-(c*p[:,0]+s*p[:,1])*scale)).astype(int)
        rain = frame['rain']
        time = frame['time']
        screen.fill((int(17-5*rain),int(32-8*rain),int(32+4*rain)))
        origin = np.floor(pos/10)*10
        for i in range(-14,15):
            for pair in ([[origin[0]+i*10,origin[1]-140],[origin[0]+i*10,origin[1]+140]],
                         [[origin[0]-140,origin[1]+i*10],[origin[0]+140,origin[1]+i*10]]):
                pygame.draw.line(screen,(25,44,47),*project(pair),1)
        center = np.asarray(frame['center'])
        normals = np.column_stack((-np.sin(center[:,2]),np.cos(center[:,2])))
        left_world = center[:,:2]+normals*frame['width']/2
        right_world = center[:,:2]-normals*frame['width']/2
        left,right = project(left_world),project(right_world)
        pygame.draw.polygon(screen,(44,54+int(7*rain),65+int(16*rain)),np.concatenate((left,right[::-1])))
        pygame.draw.lines(screen,(186,204,209),False,left,2)
        pygame.draw.lines(screen,(186,204,209),False,right,2)
        if rain > .1:
            patch_key = (frame['seed'],center[0,3],center[-1,3],frame['width'])
            if patch_key != self._patch_key:
                self._patches = wet_patches(center,frame['width'],frame['seed'])
                self._patch_key = patch_key
            for patch in self._patches.values():
                pygame.draw.polygon(screen,(49,67,82),project(patch))
        cp = project(center[:,:2])
        for i in range(len(cp)-1):
            if int(np.floor(center[i,3]/2))%2 == 0:
                pygame.draw.line(screen,(102,120,126),cp[i],cp[i+1],1)
                pygame.draw.line(screen,(184,84,79),left[i],left[i+1],3)
                pygame.draw.line(screen,(184,84,79),right[i],right[i+1],3)
            if int(center[i,3]//frame['sector_length']) != int(center[i+1,3]//frame['sector_length']):
                pygame.draw.line(screen,(232,206,129),left[i],right[i],3)
                self.text(f"S{int(center[i+1,3]//frame['sector_length'])}",cp[i,0]+10,cp[i,1],(232,206,129),self.small)
        pygame.draw.line(screen,(247,192,88),left[-1],right[-1],4)
        pygame.draw.line(screen,(222,99,114),left[0],right[0],4)
        for i in range(count):
            if ui.get('trails',True) and len(self.trails[i])>1:
                pygame.draw.lines(screen,tuple(int(x*.6) for x in COLORS[i]),False,project(self.trails[i]),2)
        if ui['lidar']:
            angles = LIDAR_ANGLES+states[follow,4]
            scan = np.asarray(frame['lidar'][follow])
            ends = pos+scan[:,None]*np.column_stack((np.cos(angles),np.sin(angles)))
            start = project([pos])[0]
            for ray,end in zip(scan,project(ends)):
                color = (249,115,102) if ray <= .11 else (52,116,137)
                pygame.draw.line(screen,color,start,end,1)
                pygame.draw.circle(screen,color,end,2)
        for i in range(count):
            if frame['status'][i] != 1:
                continue
            car = states[i]
            cy,sy = np.cos(car[4]),np.sin(car[4])
            rotation = np.array([[cy,-sy],[sy,cy]])
            visual = max(1.,15./scale)
            body = np.array([[.4,0],[.18,-.21],[-.32,-.21],[-.32,.21],[.18,.21]])*visual
            polygon = project(body@rotation.T+car[:2])
            pygame.draw.polygon(screen,(4,9,13),polygon+[3,4])
            pygame.draw.polygon(screen,COLORS[i],polygon)
            dot = project([car[:2]])[0]
            self.text(str(i+1),dot[0]+12,dot[1]-8,COLORS[i],self.small)
            if i == follow: pygame.draw.circle(screen,COLORS[i],dot,15,1)
            if rain > .2:
                tail = project([car[:2]-.7*np.array([cy,sy])])[0]
                pygame.draw.circle(screen,(109,149,169),tail,int(3+rain*3),1)
            if frame['wall_hits'][i] or frame['vehicle_hits'][i]:
                pygame.draw.circle(screen,(255,144,80),dot,22,2)
                for a in np.arange(6)*np.pi/3+time*8:
                    pygame.draw.line(screen,(255,210,123),dot+18*np.array([np.cos(a),np.sin(a)]),
                                     dot+28*np.array([np.cos(a),np.sin(a)]),2)
        # Weather overlay precedes the HUD so data remains legible.
        if rain > .05:
            veil = pygame.Surface((width,height),pygame.SRCALPHA)
            veil.fill((64,85,111,int(25*rain)))
            screen.blit(veil,(0,0))
            if self._rain_seed != frame['seed']:
                self._rain_field = RainField(frame['seed'])
                self._rain_seed = frame['seed']
            starts,ends,brightness = self._rain_field.segments(time,(width,height),rain)
            for start,end,shade in zip(starts,ends,brightness):
                color = tuple(int(value*shade) for value in (110,158,189))
                pygame.draw.line(screen,color,start,end,1)
            if rain > .7 and time%13 < .1:
                flash = pygame.Surface((width,height),pygame.SRCALPHA)
                flash.fill((185,210,231,24)); screen.blit(flash,(0,0))
        self.panel((16,16,width-32,68))
        self.text('CRASH / LEARN',34,29,TEXT,self.big)
        self.text('PROCEDURAL GRAND PRIX',36,61,MUTED,self.small)
        mode = 'REPLAY' if ui['replay'] else 'LIVE'
        self.text(f"{mode}  /  x{ui['speed']}  (reel x{ui['actual_speed']:.1f})",width*.36,31,COLORS[0])
        self.text(f"{time:07.2f}s  |  SEED {frame['seed']}  |  {ui['fps']:.0f} FPS",width*.36,56,MUTED,self.small)
        self.weather_icon(width-210,47,rain,time)
        self.text(frame['weather'],width-169,31,(175,203,234))
        self.text(f"{frame['weather_mode']} / {frame.get('weather_change_in',0.):.0f}s",width-169,56,MUTED,self.small)
        if ui['overview']:
            self.text('VUE LARGE / TAB : suivi voiture',320 if ui.get('infos',True) else 32,98,MUTED,self.small)
        if ui.get('infos',True):
            self.panel((16,98,278,344))
            self.text(f'VOITURE {follow+1} / SUIVIE',32,112,COLORS[follow])
            self.text(f'{max(0.,states[follow,3])*3.6:05.1f}',31,140,TEXT,self.huge)
            self.text('km/h',185,157,MUTED)
            rank = frame['order'].index(follow)+1
            self.text(f'POSITION {rank}/{count}',32,193,TEXT)
            self.text(f"DISTANCE {frame['distances'][follow]:.1f} m",32,221,MUTED)
            sector = frame['sectors'][follow]
            self.text(f'SECTEUR {sector+1} / {frame["sector_length"]:.0f} m',32,249,TEXT)
            fraction = (max(0.,frame['distances'][follow])%frame['sector_length'])/frame['sector_length']
            pygame.draw.rect(screen,(35,54,63),(32,276,246,5),border_radius=2)
            pygame.draw.rect(screen,COLORS[follow],(32,276,int(246*fraction),5),border_radius=2)
            times = frame['sector_times'][follow]
            self.text(f"En cours  {time-frame['sector_start'][follow]:6.2f}s",32,293,MUTED)
            self.text(f'Dernier   {times[-1]:6.2f}s' if times else 'Dernier       --',32,320,MUTED)
            self.text(f'Meilleur  {min(times):6.2f}s' if times else 'Meilleur      --',32,347,MUTED)
            finish = frame['finish_distance']
            self.text(f'Objectif {finish:.0f} m' if finish else 'COURSE SANS FIN',32,377,COLORS[0],self.small)
            self.text(f"Contacts : {frame['contacts'][follow]}",32,407,MUTED,self.small)
            self.panel((16,455,278,150))
            self.text('ADHERENCE / CAPTEURS',32,468,TEXT)
            self.text(f"mu {frame['friction']:.2f}  cible {frame['target_friction']:.2f}",32,498,MUTED)
            pygame.draw.rect(screen,(35,54,63),(32,528,246,7))
            pygame.draw.rect(screen,(101,190,224),(32,528,int(246*frame['friction']),7))
            self.text(f"Bruit lidar +/-{frame['noise']*100:.1f}%",32,547,MUTED,self.small)
            self.text(f"Pertes {frame['dropouts']*100:.2f}% / 100 rayons",32,573,MUTED,self.small)
            rx = width-290
            self.panel((rx,98,274,75+count*63))
            self.text('CLASSEMENT',rx+16,112,TEXT)
            leader = max(frame['best'])
            for rank,i in enumerate(frame['order'],1):
                yy = 146+(rank-1)*63
                self.text(f'{rank}   VOITURE {i+1}',rx+16,yy,COLORS[i])
                status = frame['status'][i]
                label = f"{frame['distances'][i]:.0f} m / -{leader-frame['best'][i]:.1f} m" if status == 1 else ('ARRIVEE' if status == 2 else 'DNF : '+frame['reasons'][i])
                self.text(label,rx+16,yy+25,MUTED,self.small)
            ey = 194+count*63
            self.panel((rx,ey,274,180))
            self.text('EVENEMENTS',rx+16,ey+12,TEXT)
            for i,event in enumerate(frame['events'][-5:]):
                self.text(f"{event['time']:5.1f} {event['text']}"[:29],rx+12,ey+43+i*25,MUTED,self.small)
            # Overview minimap is independent of chase camera rotation.
            my = min(height-185,620)
            if my >= 610:
                self.panel((16,my,278,116))
                self.text(f"CARTE / {frame['profile']}",32,my+10,MUTED,self.small)
                points = center[:,:2]
                lo,hi = points.min(axis=0),points.max(axis=0)
                mini_scale = min(236/max(1.,hi[0]-lo[0]),67/max(1.,hi[1]-lo[1]))
                def mini(p): return (np.asarray(p)-lo)*[mini_scale,-mini_scale]+[32,my+100]
                pygame.draw.lines(screen,(110,146,154),False,mini(points).astype(int),2)
                for i in range(count):
                    if frame['status'][i] == 1: pygame.draw.circle(screen,COLORS[i],mini(states[i,:2]).astype(int),3)
        self._draw_controls(frame,ui,width,height)
        if ui['paused'] or frame['terminated'] or frame['truncated']:
            message = 'PAUSE' if ui['paused'] else 'COURSE TERMINEE / REDEPART'
            self.panel((width//2-195,height//2-25,390,53))
            self.text(message,width//2-177,height//2-8,COLORS[0])

    def _draw_controls(self,frame,ui,width,height):
        self.buttons.clear()
        by = height-91
        items = [('pause','REPRENDRE' if ui['paused'] else 'PAUSE',108),('slower','- VITESSE',99),
                 ('faster','+ VITESSE',99),('follow','CAMERA [C]',110),('weather','METEO [W]',108),
                 ('profile','CARTE [P]',106),('record','STOP REC' if ui['recording'] else 'REC [E]',106),
                 ('replay','LIVE [V]' if ui['replay'] else 'REPLAY [V]',110),
                 ('infos','INFOS : OUI' if ui.get('infos',True) else 'INFOS : NON',124)]
        bx = 18
        for name,label,bw in items:
            self.button(name,label,(bx,by,bw,36),(name == 'record' and ui['recording']) or (name == 'infos' and ui.get('infos',True)))
            bx += bw+8
        self.text('ESPACE pause  +/- x1..10  TAB vue large  L lidar  T traces  R course  C voiture  E enregistrer  V replay  F11 plein ecran',20,height-43,MUTED,self.small)
        self.text('REPLAY : gauche/droite = 1 s, shift = 10 s, debut/fin ; lecture en boucle' if ui['replay'] else
                  f"Route {len(frame['center'])*.5:.0f} m / +{frame['generated']} -{frame['removed']} segments | {ui['message']}",
                  20,height-23,MUTED,self.small)
