from collections import deque
from typing import Deque, Dict, List


class ConversationalMemory:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.history.append({"user": user_message, "assistant": assistant_message})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def render_history(self) -> str:
        segments: List[str] = []
        for turn in self.history:
            segments.append(f"User: {turn['user']}")
            segments.append(f"Assistant: {turn['assistant']}")
        return "\n".join(segments)

    def clear(self) -> None:
        self.history.clear()
