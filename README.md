# samantha

# Agentic AI on Raspberry Pi (Pi5 Brain + Pi3 Client)

This repo contains:
- `brain/` (Pi 5): WebSocket server that performs STT → LLM (with tool calling) → TTS using OpenAI.
- `client/` (Pi 3): Captures microphone audio, runs VAD, sends utterances to the brain, and plays back TTS.

## Features implemented
- Wake mode: detects “Samantha” or “Krishna” from STT and selects persona.
- Query mode: user speaks request; brain replies via OpenAI TTS.
- Tool calling: Meross MSG100 garage control (left/right).

## Quick Start

### Brain (Pi5)
```bash
cd brain
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill OPENAI_API_KEY, MEROSS_EMAIL, MEROSS_PASSWORD
python -m src.brain_server
