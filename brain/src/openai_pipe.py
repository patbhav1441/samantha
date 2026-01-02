import io
import wave
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from openai import OpenAI

from . import tools as brain_tools


@dataclass
class BrainConfig:
    stt_model: str
    chat_model: str
    tts_model: str
    tts_voice: str
    system_prompt: str


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "garage_open",
            "description": "Open the user's Meross MSG100 garage door. Can target left or right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which door: 'left' or 'right'. If omitted uses first discovered.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "garage_close",
            "description": "Close the user's Meross MSG100 garage door. Can target left or right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which door: 'left' or 'right'. If omitted uses first discovered.",
                    }
                },
                "required": [],
            },
        },
    },
]


class OpenAIPipeline:
    def __init__(self, api_key: str, cfg: BrainConfig):
        self.client = OpenAI(api_key=api_key)
        self.cfg = cfg

    @staticmethod
    def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int, channels: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16)
        return buf.getvalue()

    def stt(self, pcm16: bytes, sample_rate: int, channels: int, language: Optional[str] = None) -> str:
        wav_bytes = self.pcm16_to_wav_bytes(pcm16, sample_rate, channels)
        f = io.BytesIO(wav_bytes)
        f.name = "audio.wav"

        resp = self.client.audio.transcriptions.create(
            model=self.cfg.stt_model,
            file=f,
            language=language,
        )
        return (resp.text or "").strip()

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        door = args.get("door")
        if name == "garage_open":
            return brain_tools.tool_garage_open(door=door)
        if name == "garage_close":
            return brain_tools.tool_garage_close(door=door)
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def llm(self, text: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.cfg.system_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ]

        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return (msg.content or "").strip()

        import json as pyjson

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args_dict = pyjson.loads(raw_args)
            except Exception:
                args_dict = {}

            result = self._call_tool(fn_name, args_dict)

            messages.append(
                {
                    "role": "tool",
                    "name": fn_name,
                    "tool_call_id": tc.id,
                    "content": pyjson.dumps(result),
                }
            )

        resp2 = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
        )
        final_msg = resp2.choices[0].message
        return (final_msg.content or "").strip()

    def tts_pcm24k(self, text: str) -> bytes:
        resp = self.client.audio.speech.create(
            model=self.cfg.tts_model,
            voice=self.cfg.tts_voice,
            input=text,
            response_format="pcm",
        )

        # Critical fix: openai-python returns HttpxBinaryResponseContent here in your env.
        if hasattr(resp, "read"):
            return resp.read()
        if isinstance(resp, (bytes, bytearray)):
            return bytes(resp)
        return bytes(resp)

