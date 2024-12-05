from pydantic import BaseModel


class Prompt(BaseModel):
    input: str
    max_length: int = 100
