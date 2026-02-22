"""Global hotkey registration for push-to-talk using macOS NSEvent monitors."""

import logging
from typing import Callable, Optional

from AppKit import NSEvent, NSFlagsChanged

logger = logging.getLogger(__name__)

# Right Option key on macOS
_RIGHT_OPTION_KEYCODE = 61

_monitor: Optional[object] = None


def register_ptt_hotkey(callback: Callable[[], None]) -> None:
    """Register a global monitor that fires *callback* on right Option key press.

    Uses NSEvent.addGlobalMonitorForEventsMatchingMask_handler_ which requires
    Accessibility permissions. If the monitor cannot be created (permissions not
    granted), a warning is logged but the app continues to work via the menu bar
    button.
    """
    global _monitor

    mask = NSFlagsChanged

    def _handler(event):
        if event.keyCode() == _RIGHT_OPTION_KEYCODE:
            # Only fire on key-down (modifier flag set), not key-up
            if event.modifierFlags() & (1 << 19):  # NSAlternateKeyMask bit
                logger.debug("Right Option key pressed — triggering PTT")
                callback()

    _monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, _handler)

    if _monitor is None:
        logger.warning(
            "Could not register global hotkey monitor. "
            "Grant Accessibility permission in System Settings → "
            "Privacy & Security → Accessibility for this app."
        )
    else:
        logger.info("Global PTT hotkey registered (right Option key)")


def unregister_ptt_hotkey() -> None:
    """Remove the global event monitor if one was registered."""
    global _monitor
    if _monitor is not None:
        NSEvent.removeMonitor_(_monitor)
        _monitor = None
        logger.info("Global PTT hotkey unregistered")
