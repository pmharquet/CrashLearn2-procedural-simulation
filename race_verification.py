"""Packaged-runtime smoke test, also exercised without a Python installation."""
import json
from argparse import Namespace
from pathlib import Path
import time
import traceback


def verify_distribution(window,model,output,screenshot=None):
    import pygame
    from procedural_live import LiveRace
    from race_competition import Competition,Driver,tracks
    from race_menu import CompetitionRenderer,ui_state
    live=None
    report={}
    try:
        track=next(path for path in tracks() if path.stem=='Monza_centerline')
        config=Competition(track=str(track),limit=1,drivers=[
            Driver('Azur',(35,174,244),str(model.resolve())),
            Driver('Ambre',(255,177,36),str(model.resolve()))])
        config.validate()
        live=LiveRace(Namespace(competition=config,record=None,speed=1))
        assert live.packet['ticks']==0
        live.set_pace(10,False)
        deadline=time.monotonic()+40
        while live.poll()['ticks']<20:
            pygame.event.pump()
            if time.monotonic()>deadline: raise TimeoutError('Le pilote ne fait pas avancer la course.')
            if live.packet['frame']['terminated']: raise RuntimeError('Les pilotes ont abandonné avant 20 pas.')
            time.sleep(.01)
        live.set_pace(1,True)
        frame=live.packet['frame']
        renderer=CompetitionRenderer(window.screen,window.gpu)
        renderer.colors=[d.color for d in config.drivers]
        ui=ui_state(); ui['camera_yaw']=frame['states'][0][4]
        renderer.draw(frame,ui)
        if screenshot:
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            if window.gpu: window.gpu.screenshot(screenshot)
            else: pygame.image.save(window.screen,str(screenshot))
        pygame.display.flip()
        report=dict(ok=True,ticks=live.packet['ticks'],status=frame['status'],
                    track=frame['track_name'],maps=len(tracks()),
                    renderer=window.gpu.name if window.gpu else 'software')
    except Exception:
        report=dict(ok=False,error=traceback.format_exc())
    finally:
        if live: live.close()
        if output:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report.get('ok',False)
