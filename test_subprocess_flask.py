# -*- coding: utf-8 -*-
import subprocess
import sys
import os

print("Testing subprocess with Flask app import...")
cmd = [sys.executable, "-c", """
import sys
sys.path.insert(0, 'd:/大模型技术资料库/自然语言/NLP-Course-Project-main')
from src.api.app import app
print('Flask app imported successfully')
"""]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)