import os
import re
import json
import asyncio
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import websockets
from loguru import logger
from agent.llm import create_chat_session, stream_ai_response
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


@app.get("/")
async def get_index():
    with open("frontend/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("\n[1] Browser connected to FastAPI WebSocket!")

    chat_session = create_chat_session()
    current_ai_task = None

    dg_stt_url = "wss://api.deepgram.com/v2/listen?model=flux-general-en"

    try:
        async with websockets.connect(
            dg_stt_url,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        ) as dg_socket:

            async def receive_from_browser():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await dg_socket.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.error(f"Browser Receive Error: {e}")

            async def process_deepgram():
                nonlocal current_ai_task
                try:
                    while True:
                        message = await dg_socket.recv()
                        data = json.loads(message)

                        if data.get("type") != "TurnInfo":
                            continue

                        event = data.get("event")

                        if event == "StartOfTurn":
                            await websocket.send_text(
                                json.dumps({"role": "clear_audio"})
                            )
                            if current_ai_task and not current_ai_task.done():
                                current_ai_task.cancel()
                            continue

                        if event == "EndOfTurn":
                            final_sentence = (data.get("transcript") or "").strip()
                            if final_sentence:
                                logger.info(
                                    f"\n[4] User finished speaking! '{final_sentence}'"
                                )

                                await websocket.send_text(
                                    json.dumps({"role": "clear_audio"})
                                )
                                await websocket.send_text(
                                    json.dumps({"role": "user", "text": final_sentence})
                                )

                                if current_ai_task and not current_ai_task.done():
                                    current_ai_task.cancel()

                                current_ai_task = asyncio.create_task(
                                    handle_ai_response(final_sentence)
                                )

                except Exception as e:
                    logger.error(f"Deepgram Error: {e}")

            async def handle_ai_response(user_text):
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
                                tts_url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
                                headers = {
                                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                                    "Content-Type": "application/json",
                                }
                                async with httpx.AsyncClient() as client:
                                    response = await client.post(
                                        tts_url,
                                        headers=headers,
                                        json={"text": sentence},
                                    )
                                    response.raise_for_status()

                                    await websocket.send_bytes(response.content)
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

            await asyncio.gather(receive_from_browser(), process_deepgram())

    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
