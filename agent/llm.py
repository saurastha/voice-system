import asyncio
import threading
from agno.agent import Agent
from agno.models.google import Gemini

from agent.tools import BookingSession, get_restaurant_info

SYSTEM_PROMPT = """You are a warm, efficient receptionist for The Winston Restaurant.

Keep every response strictly under 2 sentences. Never use markdown, emojis, or bullet points.

BOOKING WORKFLOW:
1. When a guest mentions making a booking or provides ANY booking details, immediately call `record_booking_details`.
2. Inspect the tool result:
   - Check 'SAVED DETAILS' vs 'STILL MISSING'.
   - Never ask for information that is already in 'SAVED DETAILS'.
   - Ask only for the items listed under 'STILL MISSING'.
3. Always ask whether a high chair is required if not yet specified.
4. Once all details are collected, briefly summarize them and ask for confirmation.
5. Only call `book_table` AFTER the guest confirms.

GENERAL QUESTIONS:
- Use `get_restaurant_info` to answer general questions (opening hours, specials, menu, location)."""


def create_chat_session():
    """Creates a new Agno Agent with its own isolated BookingSession tools."""
    session = BookingSession()

    # Tools available to this specific call
    tools = [
        get_restaurant_info,
        session.record_booking_details,
        session.book_table,
    ]

    return Agent(
        model=Gemini(id="gemini-3.7-flash"),
        description=SYSTEM_PROMPT,
        tools=tools,
        markdown=False,
    )


async def stream_ai_response(chat_session, user_text: str):
    """Async generator to stream chunks from the Agno Agent."""
    loop = asyncio.get_running_loop()
    text_queue = asyncio.Queue()
    stop_event = threading.Event()

    def run_agent():
        try:
            for chunk in chat_session.run(user_text, stream=True):
                if stop_event.is_set():
                    break
                text_delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text_delta and isinstance(text_delta, str):
                    loop.call_soon_threadsafe(text_queue.put_nowait, text_delta)
            loop.call_soon_threadsafe(text_queue.put_nowait, None)
        except Exception as e:
            print(f"LLM Error: {e}")
            loop.call_soon_threadsafe(text_queue.put_nowait, None)

    loop.run_in_executor(None, run_agent)

    try:
        while True:
            chunk = await text_queue.get()
            if chunk is None:
                break
            yield chunk
    except asyncio.CancelledError:
        stop_event.set()
        raise
