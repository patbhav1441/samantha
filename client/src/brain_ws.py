import json
import os
from typing import AsyncIterator, Optional, Union

import websockets
from dotenv import load_dotenv


class BrainWSClient:
    def __init__(self):
        load_dotenv()
        self.host = os.environ.get("BRAIN_HOST", "localhost")
        self.port = int(os.environ.get("BRAIN_PORT", "8765"))
        self.shared_secret = os.environ.get("SHARED_SECRET", "")
        self.client_id = os.environ.get("CLIENT_ID", "pi3-client")
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self, sr: int, ch: int):
        uri = f"ws://{self.host}:{self.port}"
        print(f"[WS] connecting to {uri}")
        self.ws = await websockets.connect(uri, max_size=30_000_000)

        hello = {"type": "hello", "client_id": self.client_id, "sr": sr, "ch": ch}
        if self.shared_secret:
            hello["secret"] = self.shared_secret

        await self.ws.send(json.dumps(hello))
        print("[WS] sent hello")

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def send_utterance(self, pcm16: bytes, mode: str, persona: Optional[str], sr: int, ch: int):
        if not self.ws:
            raise RuntimeError("WS not connected")

        await self.ws.send(
            json.dumps({"type": "utterance_start", "mode": mode, "persona": persona, "sr": sr, "ch": ch})
        )
        await self.ws.send(pcm16)
        await self.ws.send(json.dumps({"type": "utterance_end", "mode": mode, "persona": persona}))

    async def messages(self) -> AsyncIterator[Union[dict, bytes]]:
        if not self.ws:
            raise RuntimeError("WS not connected")
        async for msg in self.ws:
            if isinstance(msg, (bytes, bytearray)):
                yield bytes(msg)
            else:
                yield json.loads(msg)

