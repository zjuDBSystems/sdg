from queue import Queue, Empty
from fastapi import APIRouter
import asyncio
from sse_starlette.sse import EventSourceResponse
from dataclasses import dataclass
from enum import Enum

global_message_queue = Queue()

class EventType(Enum):
    REQUEST = "request"
    REASONING = "reasoning"
    RESPONSE = "response"

@dataclass
class EventResponse():
    event: EventType
    data: str

router = APIRouter(
    prefix="/events",
    tags=["events"],
)

async def event_generator():
    while True:
        try:
            asyncio.sleep(1)
            event = global_message_queue.get(timeout=1)
        except Empty:
            # If no message is received, continue waiting
            print("No message received, waiting...")
            await asyncio.sleep(1)
            continue
        yield {
            "event": event.event.value,
            "data": event.data
        }


@router.get("")
async def get_messages():
    return EventSourceResponse(event_generator())
