"""
Load environment variables from root .env file
This should be imported at the start of any GraphRAG script
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from root directory
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment variables from {env_file}")
else:
    print(f"⚠ Warning: .env file not found at {env_file}")

# Verify required variables are loaded
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"⚠ Warning: Missing environment variables: {', '.join(missing_vars)}")
else:
    print(f"✓ All required environment variables are set")
