"""Baseline trainer scaffold for JSCC/WITT in Fig.10."""


class BaselineTrainer:
    """Trainer interface for baseline systems."""

    def fit(self) -> dict:
        raise NotImplementedError
