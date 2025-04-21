from fastapi import FastAPI

from .poc import poc
from .event import router as events_router


app = FastAPI(
    docs_url="/api/docs",
)

app.include_router(events_router)

@app.get("/")
async def hello() -> dict[str, str]:
    return {"message": "Hello World"}

@app.post("/poc")
async def test():
    poc()
