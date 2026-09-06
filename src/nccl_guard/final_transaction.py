from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TransactionState(StrEnum):
    CREATED = "CREATED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


ALLOWED = {
    TransactionState.CREATED: {
        TransactionState.PREFLIGHT_PASSED,
        TransactionState.TERMINAL_FAILURE,
    },
    TransactionState.PREFLIGHT_PASSED: {
        TransactionState.STARTED,
        TransactionState.TERMINAL_FAILURE,
    },
    TransactionState.STARTED: {TransactionState.COMPLETED, TransactionState.TERMINAL_FAILURE},
    TransactionState.COMPLETED: set(),
    TransactionState.TERMINAL_FAILURE: set(),
}


@dataclass
class FinalTransaction:
    version: str
    state: TransactionState = TransactionState.CREATED

    def transition(self, state: TransactionState) -> None:
        if state not in ALLOWED[self.state]:
            raise ValueError(f"invalid transition {self.state} -> {state}")
        self.state = state
