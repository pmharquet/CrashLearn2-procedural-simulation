"""Desktop competition application."""

import argparse
from argparse import Namespace
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("auto", "gpu", "software"), default="auto")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--window-size", nargs=2, type=int)
    parser.add_argument(
        "--frames", type=int, default=0, help="Quitter après N images (vérification)."
    )
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--verify-agent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verify-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    mp.freeze_support()
    args = parse_args()
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    import pygame
    from crashlearn_sim.rendering.window import RaceWindow
    from crashlearn_sim.runtime import LiveRace
    from crashlearn_sim.recording import RaceReplay
    from crashlearn_sim.rendering.art import PALETTE
    from crashlearn_sim.ui.menu import RaceMenu
    from crashlearn_sim.ui.race_view import CompetitionRenderer, ui_state

    pygame.init()
    window = None
    live = None
    replay = None
    try:
        window = RaceWindow(args.window_size, args.fullscreen, args.renderer)
        pygame.display.set_caption("CrashLearn | Compétition IA")
        if args.verify_agent:
            from crashlearn_sim.competition.verification import verify_distribution

            if not verify_distribution(
                window, args.verify_agent, args.verify_output, args.screenshot
            ):
                raise SystemExit(1)
            return
        renderer = CompetitionRenderer(window.screen, window.gpu)
        menu = RaceMenu()
        ui = ui_state()
        clock = pygame.time.Clock()
        phase = "menu"
        buttons = {}
        frame = None
        result = None
        last_recording = None
        recording_dir = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CrashLearn" / "recordings"
        )
        countdown = 0.0
        loading_since = 0.0
        replay_index = 0
        replay_elapsed = 0.0
        measured = 0.0
        previous_ticks = previous_calls = 0
        frames = 0
        running = True

        def close_live():
            nonlocal live
            if live:
                live.close()
                live = None

        def close_replay():
            nonlocal replay
            if replay:
                replay.close()
                replay = None
            ui["replay"] = False

        def launch():
            nonlocal \
                live, \
                phase, \
                loading_since, \
                result, \
                ui, \
                renderer, \
                previous_ticks, \
                previous_calls, \
                measured
            menu.sync()
            menu.config.validate()
            close_replay()
            close_live()
            result = None
            ui = ui_state()
            renderer = CompetitionRenderer(window.screen, window.gpu)
            renderer.colors = [tuple(d.color) for d in menu.config.drivers]
            live = LiveRace(
                Namespace(
                    competition=menu.config, record=None, speed=1, recording_dir=recording_dir
                ),
                wait=False,
            )
            phase = "loading"
            loading_since = time.monotonic()
            previous_ticks = previous_calls = 0
            measured = 0.0

        while running:
            elapsed = min(0.15, clock.tick(120) / 1000)
            commands = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                    window.resized()
                    renderer.screen = window.screen
                elif event.type == pygame.KEYDOWN and (
                    event.key == pygame.K_F11
                    or (
                        event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                        and event.mod & pygame.KMOD_ALT
                    )
                ):
                    window.toggle_fullscreen()
                    renderer.screen = window.screen
                    renderer.gpu = window.gpu
                elif (
                    phase == "menu"
                    and menu.edit
                    and event.type in (pygame.TEXTINPUT, pygame.KEYDOWN)
                ):
                    menu.text_event(event)
                elif phase == "menu" and menu.scroll_event(event, window.screen.get_size()):
                    pass
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    commands.extend(k for k, r in buttons.items() if r.collidepoint(event.pos))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        commands.append(
                            "cancel"
                            if phase == "loading"
                            else "menu"
                            if phase in ("race", "countdown", "replay")
                            else "track"
                        )
                    elif phase in ("race", "countdown", "replay"):
                        mapping = {
                            pygame.K_SPACE: "pause",
                            pygame.K_TAB: "overview",
                            pygame.K_l: "lidar",
                            pygame.K_t: "trails",
                            pygame.K_h: "infos",
                            pygame.K_c: "follow",
                            pygame.K_w: "weather",
                            pygame.K_e: "record",
                            pygame.K_v: "replay",
                            pygame.K_r: "start",
                            pygame.K_PLUS: "faster",
                            pygame.K_EQUALS: "faster",
                            pygame.K_KP_PLUS: "faster",
                            pygame.K_UP: "faster",
                            pygame.K_MINUS: "slower",
                            pygame.K_KP_MINUS: "slower",
                            pygame.K_DOWN: "slower",
                        }
                        if event.key in mapping:
                            commands.append(mapping[event.key])
                        elif pygame.K_1 <= event.key <= pygame.K_9:
                            ui["speed"] = event.key - pygame.K_0
                        elif event.key == pygame.K_0:
                            ui["speed"] = 10
                        elif replay and event.key in (
                            pygame.K_LEFT,
                            pygame.K_RIGHT,
                            pygame.K_HOME,
                            pygame.K_END,
                        ):
                            delta = 200 if event.mod & pygame.KMOD_SHIFT else 20
                            replay_index = (
                                0
                                if event.key == pygame.K_HOME
                                else replay.count - 1
                                if event.key == pygame.K_END
                                else max(
                                    0,
                                    min(
                                        replay.count - 1,
                                        replay_index
                                        + (delta if event.key == pygame.K_RIGHT else -delta),
                                    ),
                                )
                            )
                            frame = replay.frame(replay_index)
                            replay_elapsed = 0.0
            for command in commands:
                try:
                    if command == "quit":
                        running = False
                    elif command in ("cancel", "menu"):
                        close_replay()
                        close_live()
                        phase = "menu"
                        menu.stage = "garage"
                        result = None
                    elif command == "start":
                        if menu.edit:
                            menu.commit_edit()
                        launch()
                    elif phase in ("menu", "results"):
                        if phase == "results" and command == "track":
                            phase = "menu"
                            result = None
                        menu.command(command)
                    elif phase in ("race", "countdown", "replay"):
                        if command in ("pause", "overview", "lidar", "trails", "infos"):
                            key = "paused" if command == "pause" else command
                            ui[key] = not ui[key]
                        elif command == "follow":
                            ui["follow"] = (ui["follow"] + 1) % len(frame["states"])
                        elif command == "faster":
                            ui["speed"] = min(10, ui["speed"] + 1)
                        elif command == "slower":
                            ui["speed"] = max(1, ui["speed"] - 1)
                        elif command.startswith("paint_"):
                            color = PALETTE[int(command.split("_")[1])]
                            renderer.colors[ui["follow"]] = color
                            menu.drivers[ui["follow"]].color = color
                        elif command == "replay" and phase != "countdown":
                            if replay:
                                close_replay()
                                phase = "race"
                                ui["paused"] = True
                                renderer.colors = [tuple(d.color) for d in menu.config.drivers]
                                frame = live.packet["frame"]
                            else:
                                live.set_pace(ui["speed"], True)
                                last_recording = live.stop_recording() or last_recording
                                if not last_recording:
                                    files = list(recording_dir.glob("*.sqlite"))
                                    last_recording = (
                                        max(files, key=lambda p: p.stat().st_mtime)
                                        if files
                                        else None
                                    )
                                if last_recording:
                                    replay = RaceReplay(last_recording)
                                    replay_index = 0
                                    replay_elapsed = 0.0
                                    frame = replay.frame(0)
                                    ui["replay"] = True
                                    ui["paused"] = False
                                    ui["follow"] = 0
                                    renderer.colors = [
                                        tuple(c)
                                        for c in frame.get(
                                            "colors", PALETTE[: len(frame["states"])]
                                        )
                                    ]
                                    phase = "replay"
                                else:
                                    ui["message"] = "Aucun enregistrement : appuyer sur E"
                        elif command in ("weather", "record") and not replay:
                            live.send(command)
                except Exception as error:
                    if phase in ("menu", "results", "loading"):
                        close_live()
                        phase = "menu"
                        menu.error = str(error)
                    else:
                        ui["message"] = str(error)

            if phase == "loading":
                try:
                    packet = live.poll()
                    if packet:
                        frame = packet["frame"]
                        phase = "countdown"
                        countdown = 3.0
                        ui["camera_yaw"] = frame["states"][0][4]
                    elif time.monotonic() - loading_since > 120:
                        raise TimeoutError("Chargement trop long : vérifiez les pilotes.")
                except Exception as error:
                    close_live()
                    phase = "menu"
                    menu.error = str(error).splitlines()[-1]

            if phase in ("race", "countdown", "replay"):
                try:
                    if phase == "countdown":
                        if not ui["paused"]:
                            countdown -= elapsed
                        if countdown <= 0:
                            phase = "race"
                    elif countdown > -0.7:
                        countdown -= elapsed
                    ui["countdown"] = countdown if countdown > -0.7 else None
                    if phase == "replay":
                        ui["countdown"] = None
                        if not ui["paused"]:
                            replay_elapsed += elapsed * ui["speed"]
                            while replay_elapsed >= replay.dt:
                                replay_index = min(replay.count - 1, replay_index + 1)
                                replay_elapsed -= replay.dt
                                if replay_index == replay.count - 1:
                                    ui["paused"] = True
                                    break
                        frame = replay.frame(replay_index)
                        ui["recording"] = False
                    else:
                        live.set_pace(ui["speed"], ui["paused"] or phase == "countdown")
                        packet = live.poll()
                        frame = live.visual_frame()
                        ui["recording"] = packet["recording"]
                        if packet["message"]:
                            ui["message"] = packet["message"]
                        if packet["last_recording"]:
                            last_recording = Path(packet["last_recording"])
                        measured += elapsed
                        if measured >= 0.5:
                            ui["ticks_per_second"] = (packet["ticks"] - previous_ticks) / measured
                            ui["ai_per_second"] = (packet["ai_calls"] - previous_calls) / measured
                            ui["actual_speed"] = ui["ticks_per_second"] * 0.05
                            previous_ticks = packet["ticks"]
                            previous_calls = packet["ai_calls"]
                            measured = 0.0
                        if frame["terminated"] or frame["truncated"]:
                            result = dict(frame, colors=renderer.colors.copy())
                            close_live()
                            phase = "results"
                    if frame["status"][ui["follow"]] != 1 and not frame["terminated"]:
                        ui["follow"] = next(i for i in frame["order"] if frame["status"][i] == 1)
                    yaw = frame["states"][ui["follow"]][4]
                    ui["camera_yaw"] += (
                        (yaw - ui["camera_yaw"] + np.pi) % (2 * np.pi) - np.pi
                    ) * min(1.0, elapsed * 4 * ui["speed"])
                    ui["fps"] = clock.get_fps()
                except Exception as error:
                    close_replay()
                    close_live()
                    phase = "menu"
                    menu.error = str(error).splitlines()[-1]

            if phase in ("menu", "loading", "results"):
                buttons = menu.draw(
                    window.screen,
                    window.gpu,
                    phase == "loading",
                    result if phase == "results" else None,
                )
            else:
                renderer.draw(frame, ui)
                buttons = renderer.buttons
            frames += 1
            if args.screenshot and args.frames and frames >= args.frames:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                if window.gpu:
                    window.gpu.screenshot(args.screenshot)
                else:
                    pygame.image.save(window.screen, str(args.screenshot))
            pygame.display.flip()
            if args.frames and frames >= args.frames:
                running = False
    finally:
        if replay:
            replay.close()
        if live:
            live.close()
        if window and window.gpu:
            window.gpu.release()
        pygame.quit()


if __name__ == "__main__":
    mp.freeze_support()
    main()
