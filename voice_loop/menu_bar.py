"""macOS menu bar application using rumps for voice loop status and PTT control."""

import logging
from typing import Callable, Optional

import rumps

logger = logging.getLogger(__name__)

# State icons (emoji fallback; could be replaced with .icns files in Phase 3)
STATE_ICONS = {
    "idle": "\U0001f3a4",           # Microphone
    "listening": "\U0001f534",      # Red circle (recording)
    "transcribing": "\u26a1",       # Processing
    "processing": "\U0001f4ad",     # Thinking
    "speaking": "\U0001f50a",       # Speaker
    "error": "\u26a0\ufe0f",        # Warning
    "disconnected": "\u274c",       # Disconnected
}


class VoiceLoopMenuBar(rumps.App):
    """Menu bar app showing voice loop state with PTT click trigger."""

    def __init__(
        self,
        on_ptt_click: Optional[Callable] = None,
        on_quit: Optional[Callable] = None,
    ):
        super().__init__(
            name="Sesame Voice",
            icon=None,
            title=STATE_ICONS["idle"],
            quit_button=None,  # Custom quit handler
        )
        self._on_ptt_click = on_ptt_click
        self._on_quit = on_quit
        self._state = "idle"
        self._mic_device = "System Default"
        self._gateway_status = "Disconnected"

        # Build menu items
        self._status_item = rumps.MenuItem("Status: Idle", callback=None)
        self._status_item.set_callback(None)
        self._mic_item = rumps.MenuItem(f"Mic: {self._mic_device}", callback=None)
        self._gateway_item = rumps.MenuItem(f"Gateway: {self._gateway_status}", callback=None)
        self._ptt_item = rumps.MenuItem("Push to Talk", callback=self._handle_ptt)
        self._quit_item = rumps.MenuItem("Quit", callback=self._handle_quit)

        self.menu = [
            self._status_item,
            self._mic_item,
            self._gateway_item,
            None,  # Separator
            self._ptt_item,
            None,  # Separator
            self._quit_item,
        ]

    def _handle_ptt(self, sender):
        """Handle push-to-talk menu item click."""
        if self._on_ptt_click:
            self._on_ptt_click()

    def _handle_quit(self, sender):
        """Handle quit menu item click."""
        if self._on_quit:
            self._on_quit()
        rumps.quit_application()

    def update_state(self, state: str) -> None:
        """Update the menu bar to reflect the current voice loop state."""
        self._state = state
        self.title = STATE_ICONS.get(state, STATE_ICONS["idle"])
        self._status_item.title = f"Status: {state.capitalize()}"

    def update_mic_device(self, device_name: str) -> None:
        """Update the displayed mic device name."""
        self._mic_device = device_name
        self._mic_item.title = f"Mic: {device_name}"

    def update_gateway_status(self, status: str) -> None:
        """Update the gateway connection status display."""
        self._gateway_status = status
        self._gateway_item.title = f"Gateway: {status}"
