import asyncio
import os
from collections import deque
from typing import Optional, Deque

import numpy as np
import webrtcvad
from dotenv import load_dotenv

from .brain_ws import BrainWSClient
from .audio_io import MicStream, SpeakerOut


IDLE = "IDLE"
AWAIT_QUERY = "AWAIT_QUERY"


async def main_async():
    load_dotenv()

    sample_rate = int(os.environ.get("SAMPLE_RATE", "16000"))
    vad_level = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))

    mic_device_substr = os.environ.get("MIC_DEVICE_SUBSTR", "USB PnP Sound Device")
    speaker_device_substr = os.environ.get("SPEAKER_DEVICE_SUBSTR", "snd_rpi_hifiberry")

    mic = MicStream(device_substr=mic_device_substr, target_samplerate=sample_rate, frame_ms=20)
    speaker = SpeakerOut(device_substr=speaker_device_substr)

    brain = BrainWSClient()
    vad = webrtcvad.Vad(vad_level)

    await brain.connect(sr=sample_rate, ch=1)
    mic.start()
    speaker.start()

    state = {"mode": IDLE, "current_persona": None}

    async def recv_task():
        # Use async-for so disconnect doesn't throw StopAsyncIteration into your face.
        tts_persona: Optional[str] = None
        async for msg in brain.messages():
            if isinstance(msg, (bytes, bytearray)):
                persona = tts_persona or state["current_persona"]
                speaker.play_pcm16(bytes(msg), src_sr=24000)
                continue

            t = msg.get("type")
            if t == "status":
                print(f"[STATUS] {msg.get('state')} | {msg.get('detail')}")
            elif t == "wake_result":
                persona = msg.get("persona")
                if persona:
                    print(f"[WAKE] persona={persona}")
                    state["mode"] = AWAIT_QUERY
                    state["current_persona"] = persona
                else:
                    print("[WAKE] none")
                    state["mode"] = IDLE
                    state["current_persona"] = None
            elif t == "assistant_text":
                persona = msg.get("persona")
                text = msg.get("text", "")
                print(f"[ASSISTANT/{persona}] {text}")
            elif t == "tts_start":
                tts_persona = msg.get("persona")
                print(f"[TTS] start persona={tts_persona}")
            elif t == "tts_end":
                print(f"[TTS] end persona={msg.get('persona')}")
                tts_persona = None

        print("[WS] recv_task ended (connection closed)")

    async def vad_task():
        q = mic.queue()
        frame_ms = 20
        frame_bytes_len = int(sample_rate * frame_ms / 1000) * 2

        speeching = False
        silence_frames = 0
        speech_frames: Deque[bytes] = deque()

        min_speech_frames = 5
        end_silence_frames = 10

        loop = asyncio.get_event_loop()

        while True:
            frame = await loop.run_in_executor(None, q.get)

            if len(frame) != frame_bytes_len:
                if len(frame) > frame_bytes_len:
                    frame = frame[:frame_bytes_len]
                else:
                    frame = frame + b"\x00" * (frame_bytes_len - len(frame))

            is_speech = vad.is_speech(frame, sample_rate)

            if is_speech:
                speech_frames.append(frame)
                speeching = True
                silence_frames = 0
            else:
                if speeching:
                    silence_frames += 1
                    if silence_frames >= end_silence_frames and len(speech_frames) >= min_speech_frames:
                        pcm16 = b"".join(speech_frames)
                        speech_frames.clear()
                        speeching = False
                        silence_frames = 0

                        if state["mode"] == IDLE:
                            send_mode = "wake"
                            send_persona = None
                        else:
                            send_mode = "query"
                            send_persona = state["current_persona"]

                        print(f"[VAD] utterance ready mode={send_mode} persona={send_persona} bytes={len(pcm16)}")
                        await brain.send_utterance(pcm16, mode=send_mode, persona=send_persona, sr=sample_rate, ch=1)

                        if send_mode == "query":
                            state["mode"] = IDLE
                            state["current_persona"] = None

    recv_fut = asyncio.create_task(recv_task())
    vad_fut = asyncio.create_task(vad_task())

    try:
        await asyncio.gather(recv_fut, vad_fut)
    finally:
        mic.stop()
        speaker.stop()
        await brain.close()


if __name__ == "__main__":
    asyncio.run(main_async())

