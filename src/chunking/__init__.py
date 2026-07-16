def build_chunks(*args, **kwargs):
    """Lazy import avoids a runpy warning when executing the pipeline module."""
    from .pipeline import build_chunks as implementation
    return implementation(*args, **kwargs)
