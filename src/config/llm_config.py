"""
LLM Configuration for AutoML Agent System
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# LLM Configuration
LLM_CONFIG = {
    "provider": "huggingface",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.2,
    "max_new_tokens": 512,
}


def get_llm():
    """Initialize and return LLM based on configuration"""
    if LLM_CONFIG["provider"] == "huggingface":
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_new_tokens=LLM_CONFIG["max_new_tokens"],
        )
        return ChatHuggingFace(llm=llm_endpoint)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_CONFIG['provider']}")
