"""Quick diagnostic to check .env file loading"""
import os
from pathlib import Path

env_file = Path(".env")

print("=" * 50)
print("ENV FILE DIAGNOSTIC")
print("=" * 50)

if env_file.exists():
    print(f"✓ .env file exists at: {env_file.absolute()}")
    print(f"  File size: {env_file.stat().st_size} bytes")
    print()
    print("Raw file contents (with repr to show hidden chars):")
    print("-" * 50)
    content = env_file.read_text(encoding='utf-8')
    for i, line in enumerate(content.split('\n'), 1):
        print(f"  {i}: {repr(line)}")
    print("-" * 50)
else:
    print("✗ .env file NOT FOUND")
    print(f"  Looking in: {Path('.').absolute()}")

print()
print("Loading dotenv...")
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✓ dotenv loaded")
except Exception as e:
    print(f"✗ dotenv error: {e}")

print()
print("Environment variables:")
print(f"  MT5_LOGIN = {repr(os.getenv('MT5_LOGIN'))}")
print(f"  MT5_PASSWORD = {repr(os.getenv('MT5_PASSWORD'))}")
print(f"  MT5_SERVER = {repr(os.getenv('MT5_SERVER'))}")

print()
mt5_login = os.getenv("MT5_LOGIN", "")
mt5_password = os.getenv("MT5_PASSWORD", "")
if mt5_login and mt5_password:
    print("✓ Credentials loaded successfully!")
else:
    print("✗ Credentials missing or empty")
