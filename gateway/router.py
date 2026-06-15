from config import settings


def select_model(requested_model: str | None, task_type: str | None) -> str:
    if requested_model:
        return requested_model
    task_map = {
        "classify": settings.small_model,
        "generate": settings.default_model,
        "summarize": settings.default_model,
    }
    if task_type and task_type in task_map:
        return task_map[task_type]
    return settings.default_model
