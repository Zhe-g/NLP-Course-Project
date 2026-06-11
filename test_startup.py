# -*- coding: utf-8 -*-
"""测试启动流程"""
import sys
import os
import traceback

print("Step 1: 启动脚本")
print(f"Python: {sys.executable}")
print(f"PID: {os.getpid()}")

try:
    print("\nStep 2: 设置路径")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"BASE_DIR: {BASE_DIR}")
    sys.path.insert(0, BASE_DIR)
    print(f"sys.path[0]: {sys.path[0]}")
    
    print("\nStep 3: 导入 Flask 应用")
    from src.api.app import app
    print("Flask 应用导入成功")
    
    print("\nStep 4: 启动 Flask")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
    
except Exception as e:
    print(f"\n错误: {e}")
    traceback.print_exc()