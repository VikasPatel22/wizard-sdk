"""
WizardAI SDK - setup.py
PyPI distribution configuration.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version from package
version = {}
exec(
    (Path(__file__).parent / "wizardai" / "__init__.py").read_text(encoding="utf-8"),
    version,
)
__version__ = version.get("__version__", "1.0.0")

setup(
    name="wizardai",
    version=__version__,
    author="WizardAI Contributors",
    author_email="hello@wizardai.dev",
    description=(
        "A powerful, all-in-one Python SDK for AI integration – combining "
        "conversational AI, computer vision, speech I/O, and more."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wizardai-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/wizardai-sdk/issues",
        "Documentation": "https://github.com/yourusername/wizardai-sdk#readme",
        "Source": "https://github.com/yourusername/wizardai-sdk",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.9",
    # Core dependencies (always installed)
    install_requires=[
        "requests>=2.28.0",
    ],
    # Optional feature groups
    extras_require={
        # AI backends
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.20.0"],
        "huggingface": ["requests>=2.28.0"],
        # Computer vision
        "vision": ["opencv-python>=4.7.0"],
        # Speech
        "speech": [
            "SpeechRecognition>=3.10.0",
            "pyttsx3>=2.90",
            "pyaudio>=0.2.13",
        ],
        "gtts": ["gtts>=2.3.0", "pygame>=2.4.0"],
        "whisper": ["openai-whisper>=20230918", "numpy>=1.24.0"],
        # All features
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "opencv-python>=4.7.0",
            "SpeechRecognition>=3.10.0",
            "pyttsx3>=2.90",
            "pyaudio>=0.2.13",
            "gtts>=2.3.0",
            "pygame>=2.4.0",
            "openai-whisper>=20230918",
            "numpy>=1.24.0",
        ],
        # Development extras
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "ruff>=0.1.0",
            "twine>=4.0",
            "build>=0.10",
        ],
    },
    keywords=[
        "ai", "chatbot", "conversational-ai", "openai", "anthropic",
        "speech-recognition", "text-to-speech", "computer-vision",
        "opencv", "nlp", "machine-learning", "sdk", "wizardai",
    ],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "wizardai=wizardai.cli:main",
        ],
    },
)
