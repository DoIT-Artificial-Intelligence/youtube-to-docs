import json
import re


def post_process_transcript(
    transcript: str | None, post_process_json: str | None
) -> dict[str, str | int]:
    """
    Run post-processing operations on a transcript based on a JSON specification.

    Args:
        transcript: The transcript text to process.
        post_process_json: A JSON string of {operation: value} pairs.
            Supported operations:
              - "word count": count occurrences of a word (case-insensitive).
                Value can be a single string or a list of strings.

    Returns:
        A dict mapping column names to results,
        e.g. {"Post-process: word count(apple)": 5}.
    """
    if not transcript or not post_process_json:
        return {}

    try:
        ops = json.loads(post_process_json)
    except json.JSONDecodeError:
        return {}

    if not isinstance(ops, dict):
        return {}

    results: dict[str, str | int] = {}

    for operation, value in ops.items():
        op_lower = operation.strip().lower()

        if op_lower == "word count":
            values = value if isinstance(value, list) else [value]
            for word in values:
                word_str = str(word).strip()
                if not word_str:
                    continue
                # Case-insensitive whole-word count
                count = len(
                    re.findall(
                        r"\b" + re.escape(word_str) + r"\b",
                        transcript,
                        re.IGNORECASE,
                    )
                )
                results[f"Post-process: word count({word_str})"] = count
        else:
            # Unknown operation — skip gracefully
            results[f"Post-process: unknown({operation})"] = "unsupported operation"

    return results
