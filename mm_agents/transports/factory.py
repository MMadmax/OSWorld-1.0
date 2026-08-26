"""Transport construction kept outside agents for dependency injection."""

from .openai_chat import OpenAIChatTransport


def build_openai_compatible_transport() -> OpenAIChatTransport:
    return OpenAIChatTransport()
