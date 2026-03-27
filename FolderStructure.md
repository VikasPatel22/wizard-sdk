wizardai-sdk/
├── .github/                # GitHub-specific configurations
│   └── workflows/
│       └── ci.yml          # Continuous Integration (test & lint) pipeline
├── examples/               # Usage examples for end-users
│   └── full_demo.py        # Comprehensive demo of SDK capabilities
├── tests/                  # Unit and integration tests
│   └── __init__.py
├── wizardai/               # Core Source Code
│   ├── client.py           # Main API client entry point
│   ├── models.py           # Data schemas and response objects
│   ├── utils.py            # Shared helper functions
│   └── ...                 # (Remaining source files)
├── .gitignore              # Files and folders to be ignored by Git
├── README.md               # Project documentation and getting started guide
├── setup.py                # Legacy build script for setuptools
├── pyproject.toml          # Modern build system requirements and metadata
├── requirements.txt        # Minimum dependencies for the SDK
└── requirements-full.txt   # Dependencies including optional/extra features