"""Voice Activity Detection using webrtcvad to detect speech start and silence end-of-speech."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class VAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 3,
        silence_threshold_ms: int = 1000,
        frame_duration_ms: int = 30,
    ):
        import webrtcvad

        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.silence_threshold_ms = silence_threshold_ms
        self.frame_duration_ms = frame_duration_ms
        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames = 0
        self._speech_detected = False
        self._frames_for_silence = silence_threshold_ms // frame_duration_ms

    def reset(self) -> None:
        """Reset VAD state for a new capture session."""
        self._silence_frames = 0
        self._speech_detected = False

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process one audio frame. Returns True if end-of-speech detected.

        Args:
            frame: int16 numpy array, exactly frame_duration_ms of audio

        Returns:
            True if silence has been detected for silence_threshold_ms after speech
        """
        # Ensure correct frame length
        expected_samples = self.sample_rate * self.frame_duration_ms // 1000
        if len(frame) != expected_samples:
            return False

        # webrtcvad expects bytes
        frame_bytes = frame.tobytes()

        is_speech = self._vad.is_speech(frame_bytes, self.sample_rate)

        if is_speech:
            self._speech_detected = True
            self._silence_frames = 0
        else:
            if self._speech_detected:
                self._silence_frames += 1

        # End-of-speech: speech was detected, then silence for threshold duration
        return self._speech_detected and self._silence_frames >= self._frames_for_silence

    @property
    def has_speech(self) -> bool:
        """Whether any speech has been detected in this session."""
        return self._speech_detected
