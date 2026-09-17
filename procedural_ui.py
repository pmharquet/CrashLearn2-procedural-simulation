"""Pygame race renderer. Weather graphics never alter simulation randomness."""
from collections import deque
from functools import lru_cache
import time
import numpy as np
import pygame
from procedural_simulation import LIDAR_ANGLES
from procedural_art import PALETTE, draw_car, draw_table

COLORS = list(PALETTE[:4])
TEXT = (217,233,235)
MUTED = (163,178,185)
ACCENT = (255,192,87)

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


def wet_patches(center,width,seed,existing=None):
    """Small wet areas off the center marking, keyed by absolute road station.

    Streaming away old points never reindexes these details or moves them.
    """
    result = {}
    existing = existing or {}
    start = int(np.ceil((center[0,3]+1.)/7.))
    end = int(np.floor((center[-1,3]-1.)/7.))
    for station in range(start,end+1):
        if station in existing:
            result[station] = existing[station]
            continue
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
    def __init__(self,screen,gpu=None):
        self.screen = screen
        self.gpu = gpu
        self._hud_time = 0.
        self._hud_key = None
        self._contacts = None
        self._contact_seed = None
        self._impacts = {}
        self.colors = COLORS.copy()
        self.font = pygame.font.SysFont('arial',17)
        self.small = pygame.font.SysFont('arial',14)
        self.big = pygame.font.SysFont('arial',27,bold=True)
        self.huge = pygame.font.SysFont('arial',52,bold=True)
        self._mouse = (0,0)
        self.trails = [deque(maxlen=650) for _ in range(4)]
        self.last_key = None
        self.buttons = {}
        self._canvas = None
        self._rain_seed = None
        self._rain_field = None
        self._patch_key = None
        self._patches = {}

    @lru_cache(maxsize=512)
    def _label(self,text,color,font):
        return font.render(text,True,color)

    def text(self,text,x,y,color=TEXT,font=None):
        self.screen.blit(self._label(str(text),tuple(color),font or self.font),(int(x),int(y)))

    @lru_cache(maxsize=32)
    def _panel_surface(self,size):
        w,h = size
        surface = pygame.Surface((w+16,h+16),pygame.SRCALPHA)
        pygame.draw.rect(surface,(0,0,0,55),(6,8,w,h),border_radius=16)
        pygame.draw.rect(surface,(22,31,39),(2,2,w,h),border_radius=14)
        pygame.draw.rect(surface,(59,72,81),(2,2,w,h),1,border_radius=14)
        pygame.draw.line(surface,(82,92,98),(18,3),(w-14,3))
        return surface

    def panel(self,rect):
        rect = pygame.Rect(rect)
        self.screen.blit(self._panel_surface(rect.size),(rect.x-2,rect.y-2))

    def button(self,name,label,rect,active=False):
        rect = pygame.Rect(rect)
        self.buttons[name] = rect
        hover = rect.collidepoint(self._mouse)
        color = ACCENT if active else ((66,80,90) if hover else (37,49,60))
        pygame.draw.rect(self.screen,color,rect,border_radius=8)
        if hover: pygame.draw.rect(self.screen,(230,233,226),rect,1,border_radius=8)
        font = self.small
        label_surface = self._label(label,(25,32,38) if active else TEXT,font)
        self.screen.blit(label_surface,label_surface.get_rect(center=rect.center))

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
        if self.gpu is not None:
            self._draw_gpu(frame,ui)
            return
        display = self.screen
        width,height = display.get_size()
        factor = min(1.,width/1280.,height/900.)
        self._mouse = pygame.mouse.get_pos()
        if factor >= 1.:
            self._draw_frame(frame,ui)
            return
        logical_size = (int(np.ceil(width/factor)),int(np.ceil(height/factor)))
        if self._canvas is None or self._canvas.get_size() != logical_size:
            self._canvas = pygame.Surface(logical_size)
        self._mouse = (self._mouse[0]*logical_size[0]/width,self._mouse[1]*logical_size[1]/height)
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

    def _update_trails(self,frame,states):
        count = len(states)
        key = (frame['seed'],frame['step'])
        if self.last_key != key:
            if self.last_key is None or key[0] != self.last_key[0] or key[1] < self.last_key[1]:
                for trail in self.trails: trail.clear()
            for i in range(count):
                if frame['status'][i] == 1: self.trails[i].append(states[i,:2].copy())
            self.last_key = key

    def collision_effects(self,frame):
        now = time.perf_counter()
        contacts = frame['contacts']
        if self._contact_seed != frame['seed'] or self._contacts is None or any(a<b for a,b in zip(contacts,self._contacts)):
            self._impacts.clear()
            self._contacts = [0]*len(contacts)
            self._contact_seed = frame['seed']
        for i,value in enumerate(contacts):
            if value>self._contacts[i]: self._impacts[i] = now
        self._contacts = list(contacts)
        self._impacts = {i:stamp for i,stamp in self._impacts.items() if now-stamp<.22}
        return {i:(now-stamp)/.22 for i,stamp in self._impacts.items()}

    def _draw_gpu(self,frame,ui):
        states = np.asarray(frame['states'])
        self._update_trails(frame,states)
        center = np.asarray(frame['center'])
        if frame['rain']>.1:
            patch_key = (frame['seed'],center[0,3],center[-1,3],frame['width'])
            if patch_key != self._patch_key:
                reuse = self._patches if self._patch_key and self._patch_key[0]==frame['seed'] and self._patch_key[3]==frame['width'] else None
                self._patches = wet_patches(center,frame['width'],frame['seed'],reuse)
                self._patch_key = patch_key
        if self._rain_seed != frame['seed']:
            self._rain_field = RainField(frame['seed'])
            self._rain_seed = frame['seed']
        self.gpu.draw(frame,ui,self.trails,self.colors,self.screen.get_size(),
                      self._patches if frame['rain']>.1 else {},self._rain_field,self.collision_effects(frame))
        now = time.perf_counter()
        # Rasterize UI at 30 Hz, but react immediately to every input/resize.
        key = (id(self.gpu),self.screen.get_size(),tuple(self.colors),pygame.mouse.get_pos(),
               tuple((k,ui.get(k)) for k in ('follow','infos','paused','overview','lidar','trails',
                     'recording','replay','speed','message')),frame['seed'],frame['terminated'],frame['truncated'])
        changed = key != self._hud_key or now-self._hud_time>=1/30
        if changed:
            self.screen.fill((0,0,0,0))
            self.buttons.clear()
            self._mouse = pygame.mouse.get_pos()
            display = self.screen
            w,h = display.get_size()
            factor = min(1.,w/1280.,h/900.)
            if factor<1:
                size = (int(np.ceil(w/factor)),int(np.ceil(h/factor)))
                if self._canvas is None or self._canvas.get_size()!=size:
                    self._canvas = pygame.Surface(size,pygame.SRCALPHA)
                self._canvas.fill((0,0,0,0))
                self.screen = self._canvas
                self._mouse = (self._mouse[0]*size[0]/w,self._mouse[1]*size[1]/h)
            width,height = self.screen.get_size()
            try:
                self._draw_hud(frame,ui,width,height,states,min(ui['follow'],len(states)-1),center)
                self._draw_controls(frame,ui,width,height)
                if not frame.get('competition') and (ui['paused'] or frame['terminated'] or frame['truncated']):
                    self.panel((width//2-195,height//2-25,390,53))
                    self.text('PAUSE' if ui['paused'] else 'COURSE TERMINEE / REDEPART',width//2-177,height//2-8,ACCENT)
            finally:
                self.screen = display
            if factor<1:
                pygame.transform.smoothscale(self._canvas,(w,h),display)
                sx,sy = w/width,h/height
                self.buttons = {name:pygame.Rect(round(r.x*sx),round(r.y*sy),max(1,round(r.w*sx)),max(1,round(r.h*sy)))
                                for name,r in self.buttons.items()}
            self._hud_key,self._hud_time = key,now
        self.gpu.overlay(self.screen,changed)

    def _draw_frame(self,frame,ui):
        screen = self.screen
        self.buttons.clear()
        width,height = screen.get_size()
        states = np.asarray(frame['states'])
        count = len(states)
        follow = min(ui['follow'],count-1)
        self._update_trails(frame,states)
        pos = states[follow,:2]
        center = np.asarray(frame['center'])
        yaw = ui['camera_yaw']
        c,s = np.cos(yaw),np.sin(yaw)
        scale = min(width/40,height/32)
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
        draw_table(screen,project,view_origin if ui['overview'] else pos,
                   np.hypot(width,height)/scale)
        center = np.asarray(frame['center'])
        normals = np.column_stack((-np.sin(center[:,2]),np.cos(center[:,2])))
        left_world = center[:,:2]+normals*frame['width']/2
        right_world = center[:,:2]-normals*frame['width']/2
        left,right = project(left_world),project(right_world)
        outer_left = project(left_world+normals*.38)
        outer_right = project(right_world-normals*.38)
        outline = np.concatenate((outer_left,outer_right[::-1]))
        pygame.draw.polygon(screen,(62,39,30),outline+[5,7])
        pygame.draw.polygon(screen,(128,153,151),outline)
        pygame.draw.polygon(screen,(56,65+int(7*rain),73+int(16*rain)),np.concatenate((left,right[::-1])))
        pygame.draw.lines(screen,(231,229,204),False,left,max(2,int(scale*.15)))
        pygame.draw.lines(screen,(231,229,204),False,right,max(2,int(scale*.15)))
        if rain > .1:
            patch_key = (frame['seed'],center[0,3],center[-1,3],frame['width'])
            if patch_key != self._patch_key:
                reuse = self._patches if self._patch_key and self._patch_key[0]==frame['seed'] and self._patch_key[3]==frame['width'] else None
                self._patches = wet_patches(center,frame['width'],frame['seed'],reuse)
                self._patch_key = patch_key
            for patch in self._patches.values():
                pygame.draw.polygon(screen,(49,67,82),project(patch))
        cp = project(center[:,:2])
        for i in range(len(cp)-1):
            if int(np.floor(center[i,3]/2))%2 == 0:
                pygame.draw.line(screen,(102,120,126),cp[i],cp[i+1],1)
                pygame.draw.line(screen,(232,83,57),left[i],left[i+1],max(3,int(scale*.19)))
                pygame.draw.line(screen,(232,83,57),right[i],right[i+1],max(3,int(scale*.19)))
            if int(center[i,3]//frame['sector_length']) != int(center[i+1,3]//frame['sector_length']):
                pygame.draw.line(screen,(232,206,129),left[i],right[i],3)
                self.text(f"S{int(center[i+1,3]//frame['sector_length'])}",cp[i,0]+10,cp[i,1],(232,206,129),self.small)
        pygame.draw.line(screen,(247,192,88),left[-1],right[-1],4)
        pygame.draw.line(screen,(222,99,114),left[0],right[0],4)
        for i in range(count):
            if ui.get('trails',True) and len(self.trails[i])>1:
                pygame.draw.lines(screen,tuple(int(x*.6) for x in self.colors[i]),False,project(self.trails[i]),2)
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
            dot = project([car[:2]])[0]
            direction = project([car[:2]+np.array([cy,sy])])[0]-dot
            heading = np.degrees(np.arctan2(-direction[0],-direction[1]))
            car_length = max(24.,min(78.,scale*1.5))
            if i == follow:
                pygame.draw.circle(screen,self.colors[i],dot,int(car_length*.57),1)
            draw_car(screen,dot,self.colors[i],heading,car_length)
            self.text(str(i+1),dot[0]+car_length*.4,dot[1]-8,self.colors[i],self.small)
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
        self._draw_hud(frame,ui,width,height,states,follow,center)
        self._draw_controls(frame,ui,width,height)
        if not frame.get('competition') and (ui['paused'] or frame['terminated'] or frame['truncated']):
            message = 'PAUSE' if ui['paused'] else 'COURSE TERMINEE / REDEPART'
            self.panel((width//2-195,height//2-25,390,53))
            self.text(message,width//2-177,height//2-8,self.colors[0])

    def _draw_hud(self,frame,ui,width,height,states,follow,center):
        screen = self.screen
        color = self.colors[follow]
        time = frame['time']
        count = len(states)
        self.panel((20,18,width-40,76))
        pygame.draw.rect(screen,ACCENT,(36,35,6,40),border_radius=3)
        self.text('CRASH / LEARN',54,28,TEXT,self.big)
        self.text('RADIO CONTROL  /  '+frame.get('track_name','PROCEDURAL RACING').upper(),55,62,MUTED,self.small)
        mode = 'REPLAY' if ui['replay'] else 'EN PISTE'
        pygame.draw.circle(screen,ACCENT if ui['replay'] else (125,220,162),(width//2-122,45),4)
        self.text(mode,width//2-110,34,TEXT)
        self.text(f"{time:06.1f} s  /  x{ui['speed']}   (reel x{ui['actual_speed']:.1f})",width//2-110,59,MUTED,self.small)
        self.weather_icon(width-233,53,frame['rain'],time)
        self.text(frame['weather'].upper(),width-193,31,TEXT)
        self.text(f"{frame['weather_mode']} / {frame.get('weather_change_in',0.):.0f}s",width-193,58,MUTED,self.small)
        if ui.get('infos',True):
            # Compact telemetry leaves the middle of the table unobstructed.
            x,y,w = 20,112,250
            self.panel((x,y,w,296))
            self.text(frame.get('names', [f'VOITURE {i+1:02d}' for i in range(count)])[follow][:12],x+18,y+16,color,self.font)
            self.text('TELEMETRIE',x+140,y+19,MUTED,self.small)
            self.text(f'{max(0.,states[follow,3])*3.6:04.1f}',x+16,y+41,TEXT,self.huge)
            self.text('km/h',x+182,y+78,MUTED,self.small)
            rank = frame['order'].index(follow)+1
            pygame.draw.line(screen,(57,70,80),(x+18,y+110),(x+w-18,y+110))
            self.text(f'POSITION {rank}/{count}',x+18,y+127,TEXT)
            self.text(f"{frame['distances'][follow]:.0f} m",x+168,y+127,color)
            self.text(f"{frame.get('unit','SECTEUR')} {frame['sectors'][follow]+1}",x+18,y+164,MUTED,self.small)
            self.text(f"{time-frame['sector_start'][follow]:.2f} s",x+164,y+160,TEXT)
            fraction = max(0.,frame['distances'][follow])%frame['sector_length']/frame['sector_length']
            pygame.draw.rect(screen,(49,62,72),(x+18,y+192,214,5),border_radius=2)
            pygame.draw.rect(screen,color,(x+18,y+192,int(214*fraction),5),border_radius=2)
            times = frame['sector_times'][follow]
            self.text('Dernier / meilleur',x+18,y+211,MUTED,self.small)
            self.text(f'{times[-1]:.2f} / {min(times):.2f} s' if times else '-- / --',x+18,y+233,TEXT)
            finish = frame['finish_distance']
            self.text(f"{frame['contacts'][follow]} contacts  /  "+(f'{finish:.0f} m' if finish else 'sans fin'),x+18,y+269,MUTED,self.small)
            self.panel((x,422,w,140))
            self.text('ADHERENCE / CAPTEURS',x+18,438,TEXT,self.small)
            self.text(f"{frame['friction']:.2f}",x+18,460,ACCENT,self.big)
            self.text(f"cible {frame['target_friction']:.2f}",x+112,474,MUTED,self.small)
            pygame.draw.rect(screen,(49,62,72),(x+18,504,214,5),border_radius=2)
            pygame.draw.rect(screen,ACCENT,(x+18,504,int(214*np.clip(frame['friction'],0,1)),5),border_radius=2)
            self.text(f"Bruit {frame['noise']*100:.1f}%  /  pertes {frame['dropouts']*100:.2f}%",x+18,527,MUTED,self.small)
            self.panel((x,576,w,158))
            self.text('CARTE / '+frame['profile'].upper(),x+18,591,MUTED,self.small)
            points = center[:,:2]
            lo,hi = points.min(axis=0),points.max(axis=0)
            mini_scale = min(210/max(1.,hi[0]-lo[0]),92/max(1.,hi[1]-lo[1]))
            def mini(p): return (np.asarray(p)-(lo+hi)/2)*[mini_scale,-mini_scale]+[x+w/2,674]
            pygame.draw.lines(screen,(89,108,120),False,mini(points).astype(int),5)
            pygame.draw.lines(screen,(167,186,191),False,mini(points).astype(int),1)
            for i in range(count):
                if frame['status'][i] == 1:
                    pygame.draw.circle(screen,self.colors[i],mini(states[i,:2]).astype(int),5 if i == follow else 3)
            rx = width-300
            self.panel((rx,112,280,52+count*53))
            self.text('CLASSEMENT',rx+18,128,TEXT)
            self.text(f'{count} RC',rx+221,132,MUTED,self.small)
            leader = max(frame['best'])
            for rank,i in enumerate(frame['order'],1):
                yy = 158+(rank-1)*53
                if i == follow:
                    pygame.draw.rect(screen,(38,54,66),(rx+8,yy,264,49),border_radius=8)
                self.text(f'{rank:02d}',rx+18,yy+9,ACCENT if rank == 1 else MUTED)
                draw_car(screen,(rx+68,yy+24),self.colors[i],-90,38)
                self.text(frame.get('names', [f'VOITURE {j+1:02d}' for j in range(count)])[i][:20],rx+95,yy+3,self.colors[i],self.small)
                status = frame['status'][i]
                label = f"{frame['distances'][i]:.0f} m / -{leader-frame['best'][i]:.1f} m" if status == 1 else ('ARRIVEE' if status == 2 else 'DNF : '+frame['reasons'][i])
                self.text(label[:26],rx+95,yy+25,MUTED,self.small)
            gy = 180+count*53
            if not frame.get('competition'):
                self.panel((rx,gy,280,230))
                self.text('ATELIER RC',rx+18,gy+16,TEXT)
                self.text(f'VOITURE {follow+1:02d}',rx+175,gy+20,color,self.small)
                pygame.draw.ellipse(screen,(13,21,28),(rx+22,gy+165,100,24))
                draw_car(screen,(rx+73,gy+125),color,-14,155)
                self.text('CARROSSERIE',rx+141,gy+64,MUTED,self.small)
                for index,paint in enumerate(PALETTE):
                    rect = pygame.Rect(rx+143+(index%3)*39,gy+95+(index//3)*41,29,29)
                    self.buttons[f'paint_{index}'] = rect
                    pygame.draw.rect(screen,paint,rect,border_radius=8)
                    if tuple(color) == paint:
                        pygame.draw.rect(screen,TEXT,rect.inflate(8,8),2,border_radius=11)
                        pygame.draw.circle(screen,(22,31,39),rect.center,4)
                    elif rect.collidepoint(self._mouse):
                        pygame.draw.rect(screen,TEXT,rect.inflate(4,4),1,border_radius=9)
                self.text('C  /  changer de voiture',rx+18,gy+204,MUTED,self.small)
            ey = gy if frame.get('competition') else gy+244
            self.panel((rx,ey,280,112))
            self.text('EVENEMENTS',rx+18,ey+13,TEXT,self.small)
            events = frame['events'][-2:]
            if not events: self.text('La piste est a vous.',rx+18,ey+47,MUTED,self.small)
            for index,event in enumerate(events):
                self.text(f"{event['time']:5.1f}s  {event['text']}"[:33],rx+18,ey+44+index*25,MUTED,self.small)

    def _draw_controls(self,frame,ui,width,height):
        # Two short groups, with explicit selected/hover states and a quiet footer.
        dock_width = 1224
        bx = (width-dock_width)//2
        by = height-112
        self.panel((bx,by,dock_width,88))
        items = [('pause','REPRENDRE' if ui['paused'] else 'PAUSE',102,ui['paused']),
                 ('slower','- VITESSE',83,False),('faster','+ VITESSE',83,False),
                 ('follow','VOITURE [C]',100,False),('overview','VUE [TAB]',91,ui['overview']),
                 ('lidar','LIDAR [L]',82,ui['lidar']),('trails','TRACES [T]',90,ui.get('trails',True)),
                 ('weather','METEO [W]',94,False),
                 ('menu' if frame.get('competition') else 'profile','ARRETER [ESC]' if frame.get('competition') else 'CIRCUIT [P]',94,False),
                 ('record','STOP REC' if ui['recording'] else 'REC [E]',80,ui['recording']),
                 ('replay','LIVE [V]' if ui['replay'] else 'REPLAY [V]',90,ui['replay']),
                 ('infos','INFOS [H]',91,ui.get('infos',True))]
        xx = bx+14
        for name,label,bw,active in items:
            self.button(name,label,(xx,by+12,bw,34),active)
            xx += bw+8
        self.text('ESPACE  pause     R  nouvelle course     F11  plein ecran',bx+18,by+60,MUTED,self.small)
        detail = ('REPLAY : fleches +/-1 s, shift 10 s, debut/fin' if ui['replay'] else ui['message'])
        self.text(detail[:48],bx+490,by+60,MUTED,self.small)
        self.text(f"{ui['fps']:.0f} FPS | {ui.get('ticks_per_second',0):.0f} ticks/s | {ui.get('ai_per_second',0):.0f} IA/s",bx+dock_width-310,by+60,ACCENT,self.small)
