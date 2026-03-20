import json


def to_json_safe(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def print_result(result):
    print(json.dumps(result, indent=2, default=to_json_safe, ensure_ascii=False))
