from fastapi import FastAPI

from .poc import data_evaluation, run_echart_task
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

@app.post("/data-evaluation-poc")
def poc2():
    data_evaluation()