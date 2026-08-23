import re
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

from agent.llm import create_chat_session, stream_ai_response
from agent.vad import SileroVAD
from agent.stt import FasterWhisperSTT
from agent.tts import KokoroTTS
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

vad_engine = SileroVAD(sample_rate=16000, silence_timeout_ms=800)
stt_engine = FasterWhisperSTT(model_size="base.en")
tts_engine = KokoroTTS(voice="af_bella")


@app.get("/")
async def get_index():
    with open("frontend/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("\n[1] Browser connected!")

    chat_session = create_chat_session()
    current_ai_task = None

    try:
        while True:
            data = await websocket.receive_bytes()
            events = vad_engine.process_audio(data)

            for event_type, speech_data in events:
                if event_type == "StartOfTurn":
                    logger.info("[2] User started speaking...")
                    await websocket.send_text(json.dumps({"role": "clear_audio"}))

                    if current_ai_task and not current_ai_task.done():
                        current_ai_task.cancel()
                        logger.info("Cancelled ongoing AI Task.")

                elif event_type == "EndOfTurn":
                    logger.info(
                        f"[3] User finished speaking! Captured {len(speech_data)} bytes."
                    )

                    final_sentence = await asyncio.to_thread(
                        stt_engine.transcribe, speech_data
                    )

                    if not final_sentence:
                        continue

                    logger.info(f"[4] Transcribed: '{final_sentence}'")

                    await websocket.send_text(json.dumps({"role": "clear_audio"}))
                    await websocket.send_text(
                        json.dumps({"role": "user", "text": final_sentence})
                    )

                    current_ai_task = asyncio.create_task(
                        handle_ai_response(final_sentence, websocket, chat_session)
                    )

    except WebSocketDisconnect:
        logger.info("Browser disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")


async def handle_ai_response(user_text, websocket, chat_session):
    tts_task = None
    try:
        sentence_queue = asyncio.Queue()

        async def tts_worker():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                try:
                    logger.info(f"[7] Generating TTS for: '{sentence}'")

                    wav_bytes = await asyncio.to_thread(
                        tts_engine.generate_audio, sentence
                    )
                    if wav_bytes:
                        await websocket.send_bytes(wav_bytes)

                except Exception as e:
                    logger.error(f"TTS Error: {e}")
                finally:
                    sentence_queue.task_done()

        tts_task = asyncio.create_task(tts_worker())
        buffer = ""

        async for text_chunk in stream_ai_response(chat_session, user_text):
            buffer += text_chunk
            await websocket.send_text(
                json.dumps({"role": "ai_chunk", "text": text_chunk})
            )

            match = re.search(r"(.*[.!?])(.*)", buffer, flags=re.DOTALL)
            if match:
                complete_sentences = match.group(1)
                buffer = match.group(2)

                sentences = re.split(r"(?<=[.!?])\s+", complete_sentences)
                for sentence in sentences:
                    if sentence.strip():
                        await sentence_queue.put(sentence.strip())

        if buffer.strip():
            await sentence_queue.put(buffer.strip())

        await sentence_queue.put(None)
        await tts_task

    except asyncio.CancelledError:
        if tts_task:
            tts_task.cancel()
        raise
