"""Compatibility export for the frozen legacy COFFEE-DAC implementation."""

from importlib import import_module

_implementation = import_module(
    "archives.uncorrected_p_pipeline_2026_08_26.coffee_dac_pipeline"
)
for _name, _value in vars(_implementation).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

__all__ = [name for name in vars(_implementation) if not name.startswith("__")]
