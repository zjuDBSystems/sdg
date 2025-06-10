from queue import Queue, Empty
from fastapi import APIRouter
import asyncio
from sse_starlette.sse import EventSourceResponse
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

global_message_queue = Queue()

logging.basicConfig(
    filename='poc.log',
    level=logging.INFO,
    format='%(message)s',
)

class EventType(Enum):
    REQUEST = "request"
    REASONING = "reasoning"
    RESPONSE = "response"

@dataclass
class EventResponse():
    event: EventType
    data: str

    def __init__(self, event, data):
        self.event = event
        self.data = data
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

router = APIRouter(
    prefix="/events",
    tags=["events"],
)

async def event_generator():
    while True:
        try:
            event = global_message_queue.get(timeout=1)
        except Empty:
            # If no message is received, continue waiting
            print("No message received, waiting...")
            await asyncio.sleep(1)
            continue
        logging.info(f"time={event.time}, event={event.event.value}, data={event.data}")
        yield {
            "event": event.event.value,
            "data": event.data,
            "comment": event.time
        }


@router.get("")
async def get_events():
    return EventSourceResponse(event_generator())
