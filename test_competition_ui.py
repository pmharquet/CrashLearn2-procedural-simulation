"""Opt-in native UI journey: CRASHLEARN_GPU_TESTS=1 python -m unittest test_competition_ui."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('CRASHLEARN_GPU_TESTS')=='1','Requires a native OpenGL display')
class CompetitionUiTests(unittest.TestCase):
    def test_paddock_countdown_race_results_restart_return(self):
        import pygame
        import crashlearn
        from race_menu import RaceMenu, CompetitionRenderer
        from procedural_live import LiveRace
        from procedural_window import RaceWindow
        output=Path('logs/qa'); output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pilote extérieur ') as directory:
            pilot=Path(directory)/'agent.py'
            pilot.write_text('class Agent:\n    def predict(self,obs,info): return 3.,0.\n')
            state=dict(stage=0,menu=None,live=None,window=None,start=time.monotonic(),frames=0)
            menu_draw=RaceMenu.draw
            race_draw=CompetitionRenderer.draw
            window_init=RaceWindow.__init__
            live_init=LiveRace.__init__
            event_get=pygame.event.get

            def capture(name):
                state['window'].gpu.screenshot(output/name)

            def window_start(window,*args,**kwargs):
                window_init(window,*args,**kwargs); state['window']=window

            def live_start(live,*args,**kwargs):
                live_init(live,*args,**kwargs); state['live']=live

            def menu_render(menu,screen,gpu=None,loading=False,result=None):
                state['menu']=menu
                buttons=menu_draw(menu,screen,gpu,loading,result)
                state['buttons']=buttons
                if menu.error: self.fail(menu.error)
                if state['stage']==0:
                    capture('selection.png')
                    state['stage']=1
                elif state['stage']==2 and menu.stage=='garage':
                    capture('paddock.png'); state['stage']=3
                elif result is not None and state['stage']==5:
                    capture('resultats.png'); state['stage']=6
                elif state['stage']==8 and not loading:
                    state['stage']=9
                return buttons

            def race_render(renderer,frame,ui):
                race_draw(renderer,frame,ui)
                if state['stage']==4 and ui.get('countdown',0)>0:
                    self.assertEqual(frame['step'],0)
                    capture('decompte.png')
                elif state['stage']==4 and frame['step']>5:
                    capture('course.png'); state['stage']=5
                elif state['stage']==7 and ui.get('countdown',0)>0:
                    self.assertEqual(frame['step'],0); state['stage']=8

            def click(name):
                return pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=state['buttons'][name].center)

            def events():
                real=event_get()
                self.assertLess(time.monotonic()-state['start'],90,'UI journey timed out')
                if state['stage']==1:
                    menu=state['menu']
                    for d in menu.drivers: d.model=str(pilot)
                    menu.command('add'); menu.command('add')
                    menu.command('name'); menu.buffer='Équipe Azur'; menu.commit_edit()
                    state['stage']=2
                    return real+[click('garage')]
                if state['stage']==3:
                    state['stage']=4
                    return real+[click('start')]
                if state['stage']==5:
                    # Accelerate the run while retaining the actual 3-second countdown.
                    return real+[pygame.event.Event(pygame.KEYDOWN,key=pygame.K_0,mod=0)]
                if state['stage']==6:
                    state['stage']=7
                    return real+[click('start')]
                if state['stage']==8:
                    return real+[pygame.event.Event(pygame.KEYDOWN,key=pygame.K_ESCAPE,mod=0)]
                if state['stage']==9:
                    self.assertEqual(state['menu'].drivers[0].name,'Équipe Azur')
                    return real+[pygame.event.Event(pygame.QUIT)]
                return real

            with patch.object(RaceWindow,'__init__',window_start), patch.object(LiveRace,'__init__',live_start), \
                 patch.object(RaceMenu,'draw',menu_render), patch.object(CompetitionRenderer,'draw',race_render), \
                 patch.object(pygame.event,'get',events), \
                 patch('sys.argv',['crashlearn.py','--renderer','gpu','--window-size','1280','900']):
                crashlearn.main()
            self.assertEqual(state['stage'],9)
            self.assertFalse(state['live'].process.is_alive())


if __name__=='__main__': unittest.main()
