"""Capture observable agent execution, never undisclosed model reasoning."""

from functools import wraps
import json


def serializable(value):
    return json.loads(
        json.dumps(value, default=str, ensure_ascii=False, allow_nan=False)
    )


def capture_agent(agent, label, task):
    return serializable(
        {
            "agent": label,
            "task": task,
            "system_prompt": agent.memory.system_prompt.system_prompt,
            "steps": agent.memory.get_full_steps(),
        }
    )


def trace_tools(tools, records, label):
    for tool in tools:
        original = tool.forward

        def wrap(fn, name):
            @wraps(fn)
            def recorded(*args, **kwargs):
                record = {
                    "agent": label,
                    "tool": name,
                    "args": serializable(args),
                    "kwargs": serializable(kwargs),
                }
                records.append(record)
                try:
                    result = fn(*args, **kwargs)
                    record["result"] = serializable(result)
                    return result
                except Exception as exc:
                    record["error"] = str(exc)
                    raise

            return recorded

        tool.forward = wrap(original, tool.name)
    return tools
