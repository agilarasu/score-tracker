"""
LLM client module (future integration).
Will send detected overlay text to an LLM for score extraction/interpretation.
"""

from typing import List, Optional


# Placeholder for future LLM integration
def send_to_llm(
    frame_number: int,
    timestamp_seconds: float,
    texts: List[str],
    model: str = None,
) -> Optional[str]:
    """
    Send detected overlay text to an LLM for processing.

    Args:
        frame_number: Frame index
        timestamp_seconds: Timestamp in seconds
        texts: Detected text strings
        model: Optional model name (e.g., gpt-4o, claude-3)

    Returns:
        LLM response string, or None if not implemented
    """
    # TODO: Implement when transitioning from txt storage to LLM
    # Example flow:
    # 1. Format: f"Frame {frame_number} @ {timestamp_seconds}s: {', '.join(texts)}"
    # 2. Call OpenAI/Anthropic/etc API
    # 3. Parse response (e.g., structured score: team_a, team_b, score_a, score_b)
    return None


def format_for_llm(
    frame_number: int,
    timestamp_seconds: float,
    texts: List[str],
) -> str:
    """
    Format detection data for LLM input.

    Args:
        frame_number: Frame index
        timestamp_seconds: Timestamp
        texts: Detected texts

    Returns:
        Formatted string suitable for LLM prompt
    """
    header = f"Frame {frame_number} @ {timestamp_seconds:.2f}s"
    content = "\n".join(texts) if texts else "(no text detected)"
    return f"{header}\n{content}"
