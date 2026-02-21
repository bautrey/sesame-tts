"""CLI entry point for the Sesame Voice Loop: python -m voice_loop."""

import argparse
import asyncio
import logging
import sys
import threading
from pathlib import Path

from .audio_capture import AudioCapture
from .config import VoiceLoopSettings
from .loop import VoiceLoop


def main():
    parser = argparse.ArgumentParser(description="Sesame Voice Loop")
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without menu bar (daemon mode)",
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = AudioCapture.list_devices()
        print("\nAudio Devices:")
        print("-" * 60)
        for dev in devices:
            direction = []
            if dev["max_input_channels"] > 0:
                direction.append(f"IN({dev['max_input_channels']}ch)")
            if dev["max_output_channels"] > 0:
                direction.append(f"OUT({dev['max_output_channels']}ch)")
            print(
                f"  [{dev['index']:2d}] {dev['name']:<40s} "
                f"{', '.join(direction)}  {dev['default_samplerate']:.0f}Hz"
            )
        print()
        sys.exit(0)

    # Load config
    config_path = Path(args.config) if args.config else None
    settings = (
        VoiceLoopSettings.from_yaml(config_path)
        if config_path
        else VoiceLoopSettings.from_yaml()
    )

    # Configure logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if settings.log_file:
        log_path = Path(settings.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    # Create voice loop
    voice_loop = VoiceLoop(settings)

    if args.headless or not settings.run_as_menu_bar:
        # Headless mode: just run the async loop
        asyncio.run(voice_loop.start())
    else:
        # Menu bar mode: run rumps on main thread, asyncio in daemon thread
        from .menu_bar import VoiceLoopMenuBar

        # We need to capture the event loop reference from the daemon thread
        loop_holder = {}

        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            loop.run_until_complete(voice_loop.start())

        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()

        # Small delay to let the async loop initialize
        import time

        time.sleep(0.1)

        def on_quit():
            loop = loop_holder.get("loop")
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(voice_loop.shutdown(), loop)

        menu_app = VoiceLoopMenuBar(
            on_ptt_click=voice_loop.trigger_ptt,
            on_quit=on_quit,
        )

        # Register state change callback to update menu bar
        voice_loop.state.on_transition(
            lambda old, new: menu_app.update_state(new.value)
        )

        menu_app.run()


if __name__ == "__main__":
    main()
