from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LOCAL_STORAGE_PATH: str = "./data"
    LOCAL_STORAGE_PATH: str = "E:\研究生\多模态数据质量评估\面向领域大模型的数据质量与制备代价评估体系\数据制备系统\sdg\data"
    # LOCAL_META_STORAGE_PATH: str = "./metadata"
    LOCAL_META_STORAGE_PATH: str = "E:\研究生\多模态数据质量评估\面向领域大模型的数据质量与制备代价评估体系\数据制备系统\sdg\metadata"
settings = Settings()
