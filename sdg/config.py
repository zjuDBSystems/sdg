from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOCAL_STORAGE_PATH: str = "./data"
    LOCAL_META_STORAGE_PATH: str = "./metadata"
    # GPT_URL: str ="https://models.inference.ai.azure.com"
    GPT_URL: str = "https://api.gptsapi.net/v1"
settings = Settings()
