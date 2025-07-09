# ───────────────── src/utils/model_introspect.py ─────────────────
from typing import Any, Dict, List
from pydantic import BaseModel
import inspect

def _field_pairs(model: BaseModel):
    """Yield (name, field_obj) compatible with v1 and v2."""
    if hasattr(model, "__fields__"):          # v1
        yield from model.__fields__.items()
    else:                                     # v2
        yield from model.model_fields.items()     # type: ignore[attr-defined]

def _field_type(f):
    """Return a readable type for the field (v1/v2)."""
    if hasattr(f, "type_"):       # v1
        return f.type_
    return f.annotation           # v2 FieldInfo

def _field_required(f) -> bool:
    if hasattr(f, "required"):            # v1
        return f.required
    return f.is_required()                # v2

def _field_default(f):
    if getattr(f, "default", None) is not None:
        return f.default
    if hasattr(f, "field_info") and getattr(f.field_info, "default", None) is not None:
        return f.field_info.default
    return None

def describe_instance(obj: Any, *, max_depth: int = 6, _depth: int = 0) -> Dict[str, Any]:
    """
    Recursively turn a Pydantic instance (or plain list/dict) into a
    dictionary that shows type, description, default, *and* current value.
    Works with Pydantic v1 **and** v2.
    """
    if _depth >= max_depth:
        return {"...": "max_depth_reached"}

    # 1️⃣ scalars ----------------------------------------------------
    if not isinstance(obj, (BaseModel, list, dict)):
        return {"value": obj}

    # 2️⃣ lists ------------------------------------------------------
    if isinstance(obj, list):
        return {
            "type": "list",
            "value": [describe_instance(i, max_depth=max_depth, _depth=_depth + 1) for i in obj]
        }

    # 3️⃣ plain dicts (unlikely after validation) -------------------
    if isinstance(obj, dict) and not isinstance(obj, BaseModel):
        return {k: describe_instance(v, max_depth=max_depth, _depth=_depth + 1) for k, v in obj.items()}

    # 4️⃣ Pydantic model --------------------------------------------
    out: Dict[str, Any] = {"type": obj.__class__.__name__}

    for name, field in _field_pairs(obj):
        raw_val = getattr(obj, name)
        info: Dict[str, Any] = {
            "type": str(_field_type(field)),
            "required": _field_required(field),
            "value": describe_instance(raw_val, max_depth=max_depth, _depth=_depth + 1)
        }

        # description
        desc = getattr(field, "description", None)
        if not desc and hasattr(field, "field_info"):
            desc = getattr(field.field_info, "description", None)
        if desc:
            info["description"] = desc

        # default
        default = _field_default(field)
        if default is not None:
            info["default"] = default

        out[name] = info

    return out
