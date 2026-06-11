# -*- coding: utf-8 -*-
import subprocess
import sys
import os

print("Testing subprocess with Python...")
result = subprocess.run([sys.executable, "-c", "print('Test successful')"], 
                       capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)