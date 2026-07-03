import config
from config import Backend, settings


def _resolve_model(requested_model: str | None, task_type: str | None) -> str:
    if requested_model:
        return requested_model
    task_map = {
        "classify": settings.small_model,
        "generate": settings.default_model,
        "summarize": settings.default_model,
        "tool": settings.tool_model,
    }
    if task_type and task_type in task_map:
        return task_map[task_type]
    return settings.default_model


def select_backend(requested_model: str | None, task_type: str | None) -> tuple[str, Backend]:
    model = _resolve_model(requested_model, task_type)
    candidates = sorted(
        [b for b in config.backends if model in b.models],
        key=lambda b: b.priority,
    )
    for backend in candidates:
        if not backend.lock.locked():
            return model, backend
    if candidates:
        return model, candidates[0]
    return model, config.backends[0]
