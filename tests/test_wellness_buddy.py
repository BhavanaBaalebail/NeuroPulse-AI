import unittest
from unittest.mock import MagicMock

from rag.llm_coach import (
    GROQ_MODEL,
    WELLNESS_BUDDY_PROMPT,
    build_wellness_buddy_messages,
    generate_wellness_buddy_reply,
)


class WellnessBuddyTests(unittest.TestCase):
    def test_no_matching_history_returns_supportive_reply(self):
        """Case (a): When no matching history exists (None or empty), provider returns supportive reply."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "I hear you. Take a deep breath, and remember to give yourself some grace."
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        conversation = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your Mental Health Wellness Buddy. "
                    "I'm here to listen, support, and console you. How are you feeling right now?"
                ),
            }
        ]
        user_question = "I feel stressed and completely drained after hours of coding."

        # Case (a) with retrieved_sessions=None
        reply = generate_wellness_buddy_reply(
            client=mock_client,
            conversation=conversation,
            user_question=user_question,
            retrieved_sessions=None,
        )

        self.assertEqual(
            reply,
            "I hear you. Take a deep breath, and remember to give yourself some grace.",
        )
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], GROQ_MODEL)
        messages = kwargs["messages"]

        # Only one system message (the friendly prompt), no evidence injected
        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]["content"], WELLNESS_BUDDY_PROMPT)
        self.assertNotIn("Optional retrieved personal-session evidence", system_messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": user_question})

    def test_zero_matching_history_empty_list_returns_supportive_reply(self):
        """Case (a) variant: Retrieval ran but returned 0 sessions (empty list)."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Take a moment to step away from the screen."
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        reply = generate_wellness_buddy_reply(
            client=mock_client,
            conversation=[],
            user_question="I'm exhausted.",
            retrieved_sessions=[],
        )

        self.assertEqual(reply, "Take a moment to step away from the screen.")
        _, kwargs = mock_client.chat.completions.create.call_args
        system_messages = [m for m in kwargs["messages"] if m["role"] == "system"]
        self.assertEqual(len(system_messages), 1)

    def test_retrieved_history_included_in_prompt_with_session_evidence(self):
        """Case (b): When history is retrieved, it is included in prompt messages as session evidence."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "In your past session sess_101, taking a 10-minute stretch break gave you an 82% recovery."
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        retrieved = [
            {
                "session_id": "sess_101",
                "date": "2026-03-15",
                "time_of_day": "afternoon",
                "fatigue_level": "moderate",
                "intervention_suggested": "desk stretch",
                "recovery_pct": 82,
                "time_to_recover_mins": 10,
            }
        ]

        reply = generate_wellness_buddy_reply(
            client=mock_client,
            conversation=[],
            user_question="What helped me before?",
            retrieved_sessions=retrieved,
        )

        self.assertEqual(
            reply,
            "In your past session sess_101, taking a 10-minute stretch break gave you an 82% recovery.",
        )
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]

        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system_messages), 2)
        evidence_content = system_messages[1]["content"]
        self.assertIn("Optional retrieved personal-session evidence", evidence_content)
        self.assertIn("sess_101", evidence_content)
        self.assertIn("2026-03-15", evidence_content)
        self.assertIn("afternoon", evidence_content)
        self.assertIn("level=moderate", evidence_content)
        self.assertIn("intervention=desk stretch", evidence_content)
        self.assertIn("recovery=82%", evidence_content)
        self.assertIn("time_to_recover=10 min", evidence_content)
        self.assertEqual(messages[-1], {"role": "user", "content": "What helped me before?"})

    def test_provider_failure_returns_supportive_fallback_without_raising(self):
        """When client throws an exception, buddy returns a non-empty contextual fallback rather than crashing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Groq upstream connection reset")

        reply = generate_wellness_buddy_reply(
            client=mock_client,
            conversation=[],
            user_question="I feel overwhelmed.",
            retrieved_sessions=None,
        )
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)
        self.assertIn("offline", reply.lower())

    def test_none_client_returns_supportive_fallback(self):
        """When client is None, buddy returns a non-empty contextual fallback rather than raising."""
        reply = generate_wellness_buddy_reply(
            client=None,
            conversation=[],
            user_question="Hello buddy",
            retrieved_sessions=None,
        )
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)
        self.assertIn("offline", reply.lower())

    def test_build_wellness_buddy_messages_limits_conversation_history(self):
        """Conversation history is limited to the most recent 8 messages."""
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
            for i in range(15)
        ]
        messages = build_wellness_buddy_messages(
            conversation=history,
            user_question="latest query",
            retrieved_sessions=None,
        )
        # 1 system + 8 history + 1 user question
        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], history[-8])
        self.assertEqual(messages[-1], {"role": "user", "content": "latest query"})


if __name__ == "__main__":
    unittest.main()

