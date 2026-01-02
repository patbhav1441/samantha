import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import websockets
from websockets import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

from .openai_pipe import OpenAIPipeline, BrainConfig


@dataclass
class UtteranceBuffer:
    pcm: bytearray
    mode: str
    persona: Optional[str]
    sample_rate: int
    channels: int
    started_at: float


class BrainServer:
    def __init__(
        self,
        pipeline: OpenAIPipeline,
        shared_secret: str = "",
        persona_prompts: Optional[Dict[str, str]] = None,
    ):
        self.pipeline = pipeline
        self.shared_secret = shared_secret.strip()
        self.buffers: Dict[WebSocketServerProtocol, Optional[UtteranceBuffer]] = {}
        self.persona_prompts = persona_prompts or {}

    async def send_status(self, ws: WebSocketServerProtocol, state: str, detail: str = ""):
        try:
            await ws.send(json.dumps({"type": "status", "state": state, "detail": detail}))
        except Exception:
            pass

    def _detect_wake_persona(self, text: str) -> Optional[str]:
        t = text.lower()
        if "samantha" in t:
            return "samantha"
        if "krishna" in t:
            return "krishna"
        return None

    def _persona_system_prompt(self, persona: Optional[str]) -> Optional[str]:
        if not persona:
            return None
        return self.persona_prompts.get(persona)

    async def handle_client(self, ws: WebSocketServerProtocol):
        client_id = "unknown"
        self.buffers[ws] = None

        try:
            raw = await ws.recv()
            hello = json.loads(raw)
            if hello.get("type") != "hello":
                await ws.close()
                return

            client_id = hello.get("client_id", "unknown")
            sr = int(hello.get("sr", 16000))
            ch = int(hello.get("ch", 1))

            if self.shared_secret and hello.get("secret", "") != self.shared_secret:
                await ws.close()
                return

            print(f"[BRAIN] client connected: {client_id} sr={sr} ch={ch}")
            await self.send_status(ws, "IDLE", "Connected")

            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    buf = self.buffers.get(ws)
                    if buf:
                        buf.pcm.extend(msg)
                    continue

                try:
                    data = json.loads(msg)
                except Exception:
                    print(f"[BRAIN] bad JSON from {client_id}: {msg!r}")
                    continue

                t = data.get("type")

                if t == "utterance_start":
                    mode = data.get("mode", "query")
                    persona = data.get("persona")
                    sr_msg = int(data.get("sr", sr))
                    ch_msg = int(data.get("ch", ch))

                    self.buffers[ws] = UtteranceBuffer(
                        pcm=bytearray(),
                        mode=mode,
                        persona=persona,
                        sample_rate=sr_msg,
                        channels=ch_msg,
                        started_at=time.time(),
                    )
                    print(f"[BRAIN] utterance_start {client_id} mode={mode} persona={persona}")
                    await self.send_status(ws, "LISTENING", "Listening…")

                elif t == "utterance_end":
                    buf = self.buffers.get(ws)
                    self.buffers[ws] = None

                    mode = data.get("mode", getattr(buf, "mode", "query"))
                    persona = data.get("persona", getattr(buf, "persona", None))

                    if not buf or len(buf.pcm) < 2000:
                        print(f"[BRAIN] utterance_end {client_id} (empty)")
                        await self.send_status(ws, "IDLE")
                        continue

                    pcm_bytes = bytes(buf.pcm)
                    print(f"[BRAIN] utterance_end {client_id} mode={mode} persona={persona} bytes={len(pcm_bytes)}")

                    await self.send_status(ws, "THINKING", "Transcribing…")
                    text = await asyncio.to_thread(
                        self.pipeline.stt,
                        pcm_bytes,
                        buf.sample_rate,
                        buf.channels,
                        "en",  # force English
                    )
                    print(f"[STT] {text!r}")

                    if not text:
                        await self.send_status(ws, "IDLE")
                        continue

                    if mode == "wake":
                        detected = self._detect_wake_persona(text)
                        await ws.send(json.dumps({"type": "wake_result", "persona": detected}))
                        await self.send_status(ws, "IDLE")
                        continue

                    await self.send_status(ws, "THINKING", "Thinking…")
                    sys_prompt = self._persona_system_prompt(persona)
                    reply = await asyncio.to_thread(self.pipeline.llm, text, sys_prompt)
                    print(f"[LLM/{persona}] {reply!r}")

                    await ws.send(json.dumps({"type": "assistant_text", "persona": persona, "text": reply}))

                    await self.send_status(ws, "SPEAKING", "Speaking…")
                    await ws.send(json.dumps({"type": "tts_start", "persona": persona}))

                    pcm24k = await asyncio.to_thread(self.pipeline.tts_pcm24k, reply)
                    await ws.send(pcm24k)

                    await ws.send(json.dumps({"type": "tts_end", "persona": persona}))
                    await self.send_status(ws, "IDLE")

        except ConnectionClosed:
            pass
        except Exception as e:
            print(f"[BRAIN] error handling client {client_id}: {e}")
        finally:
            print(f"[BRAIN] client disconnected: {client_id}")
            self.buffers.pop(ws, None)


async def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    host = os.environ.get("BRAIN_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("BRAIN_BIND_PORT", "8765"))
    secret = os.environ.get("SHARED_SECRET", "")

    cfg = BrainConfig(
        stt_model=os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        chat_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        tts_model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice=os.environ.get("OPENAI_TTS_VOICE", "coral"),
        system_prompt=os.environ.get("SYSTEM_PROMPT", "You are a helpful voice assistant."),
    )

    persona_prompts = {
        "samantha": os.environ.get("SAMANTHA_SYSTEM_PROMPT", cfg.system_prompt),
        "krishna": os.environ.get("KRISHNA_SYSTEM_PROMPT", cfg.system_prompt),
    }

    pipeline = OpenAIPipeline(api_key, cfg)
    server = BrainServer(pipeline, shared_secret=secret, persona_prompts=persona_prompts)

    print(f"[BRAIN] starting ws server on {host}:{port}")
    async with websockets.serve(server.handle_client, host, port, max_size=30_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

