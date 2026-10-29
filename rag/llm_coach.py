import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq model to use for the Wellness Buddy.
# Change this constant if the model name changes on your Groq account.
# Run:  python3 -c "from groq import Groq; c=Groq(); [print(m.id) for m in c.models.list().data]"
# to list available models.
# ---------------------------------------------------------------------------
GROQ_MODEL = "qwen/qwen3.8-27b"

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore

from rag.retriever import format_retrieved_sessions_for_prompt


SYSTEM_PROMPT = """You are NeuroPulse AI's personal fatigue coach. You have access to this user's past fatigue sessions and intervention history. Be specific, data-driven, and personal. Never give generic advice. Always cite the retrieved sessions with dates and recovery percentages. Keep response under 120 words. Format: 2-3 sentences of insight + 1 bold recommendation."""

WELLNESS_BUDDY_PROMPT = (
    "You are NeuroPulse AI's friendly Wellness Buddy. Respond naturally to the user's "
    "free-text question, be warm and supportive, and keep replies concise. Do not give "
    "medical advice or diagnose. Personal history is optional context, not a requirement "
    "for answering. If retrieved sessions are relevant, ground claims in them and cite "
    "their dates/session IDs and observed recovery. Do not claim a session was retrieved "
    "when no records were provided."
)


def build_wellness_buddy_messages(
    conversation: List[Dict[str, str]],
    user_question: str,
    retrieved_sessions: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": WELLNESS_BUDDY_PROMPT}]
    if retrieved_sessions:
        evidence = []
        for session in retrieved_sessions:
            evidence.append(
                " | ".join(
                    str(value)
                    for value in (
                        session.get("session_id") or "session",
                        session.get("date") or "date unavailable",
                        session.get("time_of_day") or "time unavailable",
                        f"level={session.get('fatigue_level') or 'unrecorded'}",
                        f"intervention={session.get('intervention_suggested') or 'unrecorded'}",
                        f"recovery={session.get('recovery_pct') if session.get('recovery_pct') is not None else 'unrecorded'}%",
                        f"time_to_recover={session.get('time_to_recover_mins') if session.get('time_to_recover_mins') is not None else 'unrecorded'} min",
                    )
                )
            )
        messages.append(
            {
                "role": "system",
                "content": "Optional retrieved personal-session evidence. Use only if relevant; cite the session ID/date and do not infer beyond these records:\n"
                + "\n".join(evidence),
            }
        )
    recent = list(conversation[-8:]) if conversation else []
    if recent and recent[-1].get("role") == "user" and recent[-1].get("content") == user_question:
        recent = recent[:-1]
    messages.extend(recent)
    messages.append({"role": "user", "content": user_question})
    return messages


def generate_contextual_fallback(
    user_question: str,
    retrieved_sessions: List[Dict[str, Any]] | None = None,
) -> str:
    """Generate an honest, varied, contextual fallback when the LLM provider is unavailable."""
    q_lower = user_question.lower()

    if retrieved_sessions and any(k in q_lower for k in ["past", "before", "helped", "history", "previous", "record"]):
        best = retrieved_sessions[0]
        intervention = best.get("intervention_suggested", "taking a short break")
        recovery = best.get("recovery_pct")
        rec_str = f"with ~{recovery}% recovery" if recovery is not None else "as recorded"
        session_id = best.get("session_id", "earlier session")
        return (
            f"Live AI coaching is currently offline, but looking at your retrieved history ({session_id}): "
            f"**{intervention}** was effective {rec_str}. Try that intervention first while taking a breather."
        )

    if any(k in q_lower for k in ["drain", "tired", "exhaust", "heavy", "burnout", "worn"]):
        return (
            "Live AI assistance is currently offline, but I hear you: feeling drained after intense work is a real signal. "
            "Step away from your screen for a 5-minute break, drink a glass of water, and give yourself a moment to decompress."
        )

    if any(k in q_lower for k in ["reset", "break", "quick", "short pause"]):
        return (
            "Live AI assistance is currently offline, but here is a quick reset: practice 60 seconds of slow 4-6 breathing, "
            "roll your shoulders back, and look at an object 20 feet away to rest your eyes. Physical micro-pauses help lower cognitive strain."
        )

    if any(k in q_lower for k in ["early", "sign", "notice", "symptom"]):
        return (
            "Live AI assistance is currently offline, but key early fatigue signs to watch for include: slower typing rhythm, "
            "frequent backspacing, glazed focus, heavier blink rates, and longer idle times between actions. Catching these early lets you pace yourself."
        )

    cleaned_q = user_question.strip().rstrip("?.!")
    if len(cleaned_q) > 40:
        cleaned_q = cleaned_q[:40] + "..."
    return (
        f"Live AI connectivity is currently offline, so I cannot generate a full live response for '{cleaned_q}'. "
        "I'm still here alongside your session though. Remember to pace yourself, stay hydrated, and take a brief break when you need it."
    )


def generate_wellness_buddy_reply(
    client: Any,
    conversation: List[Dict[str, str]],
    user_question: str,
    retrieved_sessions: List[Dict[str, Any]] | None = None,
) -> str:
    if client is None:
        return generate_contextual_fallback(user_question, retrieved_sessions)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=build_wellness_buddy_messages(
                conversation, user_question, retrieved_sessions
            ),
        )
        choice = response.choices[0]
        if hasattr(choice, "message"):
            msg = choice.message
            content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
        elif isinstance(choice, dict):
            msg = choice.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        else:
            content = str(choice)

        if content and str(content).strip():
            return str(content).strip()
        return generate_contextual_fallback(user_question, retrieved_sessions)
    except Exception as exc:
        # Log the error type and message but never the API key.
        logger.warning(
            "Groq API call failed [model=%s]: %s: %s",
            GROQ_MODEL,
            type(exc).__name__,
            str(exc)[:200],
        )
        return generate_contextual_fallback(user_question, retrieved_sessions)


def _build_user_prompt(current_state: Dict[str, Any], retrieved_sessions: List[Dict[str, Any]]) -> str:
    retrieved_sessions_formatted = format_retrieved_sessions_for_prompt(retrieved_sessions)
    return f"""Current fatigue state:
- Fatigue Level: {current_state['fatigue_level']}
- EAR: {current_state['EAR']} | MAR: {current_state['MAR']} | MCR: {current_state['MCR']}
- MPTI: {current_state['MPTI']} | ITSL: {current_state['ITSL']}ms | TRCV: {current_state['TRCV']} | DCR: {current_state['DCR']}
- Time of day: {current_state['time_of_day']}
- Session type: {current_state['session_type']}

Retrieved from your personal history:
{retrieved_sessions_formatted}

Based on this user's OWN past data, what intervention should they take
right now and why?"""


def generate_coaching_recommendation(
    current_state: Dict[str, Any],
    retrieved_sessions: List[Dict[str, Any]],
) -> str:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Build a deterministic fallback from retrieved history so the dashboard remains useful
    # even when external API calls fail (billing, quota, network, etc.).
    intervention_stats: Dict[str, List[float]] = {}
    for row in retrieved_sessions:
        key = str(row.get("intervention_suggested", "breathing exercise"))
        intervention_stats.setdefault(key, []).append(float(row.get("recovery_pct", 0)))
    if intervention_stats:
        best_name = max(intervention_stats, key=lambda k: sum(intervention_stats[k]) / max(1, len(intervention_stats[k])))
        best_avg = sum(intervention_stats[best_name]) / max(1, len(intervention_stats[best_name]))
    else:
        best_name = "breathing exercise"
        best_avg = 0.0

    fallback_text = (
        "Live LLM coaching is temporarily unavailable. "
        f"Based on your retrieved session history, **{best_name}** is the strongest immediate option "
        f"(~{best_avg:.1f}% average recovery in similar sessions)."
    )

    user_prompt = _build_user_prompt(current_state, retrieved_sessions)

    # 1) Try Claude first if key + SDK are available.
    if anthropic_api_key and Anthropic is not None:
        try:
            client = Anthropic(api_key=anthropic_api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=220,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
            if text_parts:
                return "\n".join(text_parts).strip()
        except Exception:
            pass

    # 2) Fallback to Gemini if key + SDK are available.
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_prompt = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}"
            response = model.generate_content(gemini_prompt)
            text = getattr(response, "text", "")
            if text:
                return text.strip()
        except Exception:
            pass

    return fallback_text
