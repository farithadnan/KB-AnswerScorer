from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Choice:
    index: int
    finish_reason: str
    message: Message

@dataclass
class ChatResponse:
    id: str
    created: int
    model: str
    choices: List[Choice]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    response_token_per_s: Optional[float] = None
    prompt_token_per_s: Optional[float] = None
    total_duration: Optional[int] = None
    approximate_total: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict):
        choices = [
            Choice(
                index=c["index"],
                finish_reason=c.get("finish_reason", ""),
                message=Message(
                    role=c["message"]["role"],
                    content=c["message"]["content"]
                )
            ) for c in data.get("choices", [])
        ]
        return cls(
            id=data["id"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            total_tokens=data.get("total_tokens", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            response_token_per_s=data.get("response_token/s"),
            prompt_token_per_s=data.get("prompt_token/s"),
            total_duration=data.get("total_duration"),
            approximate_total=data.get("approximate_total")
        )