"""Lesson pool for storing and managing distilled lessons."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A single lesson distilled from solution or debug experience."""

    id: str
    category: Literal["solution", "debug"]
    description: str
    source_node: str = ""


class LessonPool:
    """Pool of lessons with bounded size and serialization support."""

    def __init__(self, max_lessons: int = 30, category: str = "solution") -> None:
        self.max_lessons = max_lessons
        self.category = category
        self.lessons: list[Lesson] = []

    def add(self, lesson: Lesson) -> bool:
        """Add lesson to pool. Returns True if added, False if pool is full (evicts oldest)."""
        if len(self.lessons) >= self.max_lessons:
            self.lessons.pop(0)  # Remove oldest
        self.lessons.append(lesson)
        logger.info("Added %s lesson: %s", self.category, lesson.id)
        return True

    def format_lessons(self) -> str:
        """Format all lessons as a string for LLM context."""
        if not self.lessons:
            return "No lessons available."
        parts = []
        for i, lesson in enumerate(self.lessons, 1):
            parts.append(f"Lesson {i} (ID: {lesson.id}):\n{lesson.description}")
        return "\n\n".join(parts)

    def save(self, path: str) -> None:
        """Save lessons to a JSON file."""
        data = [{"id": lesson.id, "description": lesson.description} for lesson in self.lessons]
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str) -> None:
        """Load lessons from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.lessons = [
            Lesson(id=d["id"], category=self.category, description=d["description"])  # type: ignore[arg-type]
            for d in data
        ]
