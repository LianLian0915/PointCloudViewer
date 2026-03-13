from __future__ import annotations

import numpy as np


class HistoryStack:
    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    def push_command(self, cmd: dict) -> None:
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self, model) -> bool:
        if not self.undo_stack:
            return False
        cmd = self.undo_stack.pop()
        inverse = model.apply_inverse_command(cmd)
        if inverse is not None:
            self.redo_stack.append(inverse)
        return True

    def redo(self, model) -> bool:
        if not self.redo_stack:
            return False
        cmd = self.redo_stack.pop()
        inverse = model.apply_inverse_command(cmd)
        if inverse is not None:
            self.undo_stack.append(inverse)
        return True
