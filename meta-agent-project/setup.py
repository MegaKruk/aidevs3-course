from setuptools import setup, find_packages

setup(
    name="meta-agent",
    version="1.0.0",
    description="Autonomous LLM-powered homework solver for AI courses",
    author="Meta Agent Team",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.5.3",
        "requests>=2.31.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "meta-agent=meta_agent_main:main",
        ],
    },
)
