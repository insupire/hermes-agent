"""wedid fork: 호스트가 정체를 대면 그 글이 첫 자리에 서고, 안 대면 상위 동작 그대로다."""

from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, GOOGLE_MODEL_OPERATIONAL_GUIDANCE
from agent.system_prompt import build_system_prompt_parts
from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _agent(monkeypatch, tmp_path, **kw):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "SOUL.md").write_text("soul words", encoding="utf-8")
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    agent = AIAgent(
        api_key="k", base_url="http://test", provider="openrouter", api_mode="chat_completions",
        model="gemini-3.5-flash", max_iterations=1, quiet_mode=True, skip_memory=True,
        skip_context_files=True, enabled_toolsets=["todo"], platform="host", **kw,
    )
    agent.valid_tool_names = {"todo_list"}
    return agent


def _prompt(agent, system_message=None):
    return "\n\n".join(build_system_prompt_parts(agent, system_message).values())


def test_a_hosted_identity_takes_the_first_place(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, identity_prompt="host words", load_soul_identity=True)
    prompt = _prompt(agent, "this run")
    assert prompt.startswith("host words")
    assert "soul words" not in prompt and "Hermes Agent" not in prompt
    assert "read_file" not in prompt and "OUT-OF-BAND" not in prompt and "Active Hermes profile" not in prompt
    assert "MUST use your tools" in prompt and "**Keep going:**" in prompt
    assert "Conversation started:" in prompt
    assert prompt.index("MUST use your tools") < prompt.index("this run"), "실행별 글은 뒤 층에 남는다"


def test_without_it_the_prompt_is_the_upstream_one(monkeypatch, tmp_path):
    prompt = _prompt(_agent(monkeypatch, tmp_path))
    assert prompt.startswith(DEFAULT_AGENT_IDENTITY)
    assert "You run on Hermes Agent" in prompt and "OUT-OF-BAND" in prompt
    assert GOOGLE_MODEL_OPERATIONAL_GUIDANCE.strip() in prompt


def test_an_empty_identity_is_no_identity(monkeypatch, tmp_path):
    assert _prompt(_agent(monkeypatch, tmp_path, identity_prompt="")).startswith(DEFAULT_AGENT_IDENTITY)
