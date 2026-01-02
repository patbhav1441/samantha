import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

from .audio_utils import pcm16_to_float, float_to_pcm16, resample_float


@dataclass
class PickedDevice:
    index: int
    name: str


def pick_device_by_substring(substr: str, want_input: bool) -> PickedDevice:
    substr_l = (substr or "").lower()
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        name = (d["name"] or "")
        if substr_l in name.lower():
            if want_input and d["max_input_channels"] > 0:
                candidates.append((i, name))
            if (not want_input) and d["max_output_channels"] > 0:
                candidates.append((i, name))
    if not candidates:
        raise RuntimeError(f"No {'input' if want_input else 'output'} device matched substring: {substr!r}")
    idx, name = candidates[0]
    return PickedDevice(index=idx, name=name)


class MicStream:
    """
    Captures mic audio from hardware sample rate, resamples to target_sr,
    and emits 20ms PCM16 frames at target_sr suitable for webrtcvad.
    """
    def __init__(self, device_substr: str, target_samplerate: int = 16000, frame_ms: int = 20):
        pick = pick_device_by_substring(device_substr, want_input=True)
        self.device_index = pick.index
        self.device_name = pick.name

        dev = sd.query_devices(self.device_index, "input")
        self.hw_sr = int(dev["default_samplerate"])

        self.target_sr = int(target_samplerate)
        self.frame_ms = int(frame_ms)

        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=400)
        self._stream: Optional[sd.InputStream] = None

        # compute hw blocksize to approximate frame_ms at hw_sr
        self.hw_block = int(round(self.hw_sr * self.frame_ms / 1000))

        # internal float buffer at target sr
        self._accum = np.zeros(0, dtype=np.float32)

    def start(self):
        def cb(indata, frames, time_info, status):
            if status:
                print(f"[MIC] status: {status}")

            # indata float32 with shape (frames, channels)
            x = indata[:, 0].astype(np.float32)  # mono
            # resample hw_sr -> target_sr
            x_rs = resample_float(x, self.hw_sr, self.target_sr)
            self._accum = np.concatenate([self._accum, x_rs])

            # output exact frame_ms chunks
            frame_len = int(self.target_sr * self.frame_ms / 1000)
            while len(self._accum) >= frame_len:
                chunk = self._accum[:frame_len]
                self._accum = self._accum[frame_len:]
                pcm = float_to_pcm16(chunk)
                try:
                    self._q.put_nowait(pcm)
                except queue.Full:
                    pass

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.hw_sr,
            blocksize=self.hw_block,
            dtype="float32",
            callback=cb,
        )
        self._stream.start()
        print(f"[MIC] started on device {self.device_index} ({self.device_name}), hw_sr={self.hw_sr} Hz, target_sr={self.target_sr} Hz")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[MIC] stopped")

    def queue(self) -> "queue.Queue[bytes]":
        return self._q


class SpeakerOut:
    """
    Plays PCM16 audio. Incoming from brain is PCM16 @ 24k.
    Output device likely runs at 44100; we resample to device SR for PortAudio.
    """
    def __init__(self, device_substr: str, samplerate: Optional[int] = None, channels: int = 2, blocksize: int = 1024):
        pick = pick_device_by_substring(device_substr, want_input=False)
        self.device_index = pick.index
        self.device_name = pick.name

        dev = sd.query_devices(self.device_index, "output")
        self.device_sr = int(dev["default_samplerate"])

        self.samplerate = self.device_sr if samplerate is None else int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream: Optional[sd.OutputStream] = None

    def start(self):
        def cb(outdata, frames, time_info, status):
            outdata[:] = 0
            try:
                x = self._q.get_nowait()
            except queue.Empty:
                return

            # x is mono float32
            n = min(frames, len(x))
            if self.channels == 1:
                outdata[:n, 0] = x[:n]
            else:
                outdata[:n, 0] = x[:n]
                outdata[:n, 1] = x[:n]

        self._stream = sd.OutputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=cb,
        )
        self._stream.start()
        print(f"[SPEAKER] started on device {self.device_index} ({self.device_name}), {self.samplerate} Hz")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[SPEAKER] stopped")

    def play_pcm16(self, pcm: bytes, src_sr: int = 24000):
        x = pcm16_to_float(pcm)
        if src_sr != self.samplerate:
            x = resample_float(x, src_sr, self.samplerate)

        step = self.blocksize
        for i in range(0, len(x), step):
            chunk = x[i:i+step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)), mode="constant")
            try:
                self._q.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                pass
