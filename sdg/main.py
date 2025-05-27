from fastapi import FastAPI

from .poc import run_echart_task
from .event import router as events_router


app = FastAPI(
    docs_url="/api/docs",
)

app.include_router(events_router)

@app.get("/")
async def hello() -> dict[str, str]:
    return {"message": "Hello World"}

@app.post("/poc")
def poc1():
    run_echart_task()
