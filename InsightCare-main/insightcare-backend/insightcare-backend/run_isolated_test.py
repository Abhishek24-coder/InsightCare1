"""
Start backend server in isolated process and run tests
"""

import subprocess
import time
import sys
import os

# Change to backend directory
backend_dir = (
    r"C:\Users\WINDOWS 11\OneDrive\Desktop\healthcare web app\insightcare-backend"
)
os.chdir(backend_dir)

# Python executable from venv
python_exe = (
    r"C:\Users\WINDOWS 11\OneDrive\Desktop\healthcare web app\env\Scripts\python.exe"
)

print("Starting backend server in isolated process...")
print("=" * 60)

# Start uvicorn in separate process with CREATE_NEW_CONSOLE flag
server_process = subprocess.Popen(
    [
        python_exe,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ],
    creationflags=subprocess.CREATE_NEW_CONSOLE,  # New console window
    cwd=backend_dir,
)

print(f"Backend server started (PID: {server_process.pid})")
print("Waiting 5 seconds for server to initialize...")
time.sleep(5)

# Check if server is still running
if server_process.poll() is not None:
    print(f"❌ SERVER CRASHED IMMEDIATELY (exit code: {server_process.returncode})")
    sys.exit(1)

print("✅ Server is running in new console window")
print("\nNow running endpoint tests...\n")

# Run the test script
test_result = subprocess.run(
    [python_exe, "test_diagnose_endpoint.py"], cwd=backend_dir, capture_output=False
)

print("\n" + "=" * 60)
print("Test completed!")
print(f"Backend server still running: {server_process.poll() is None}")
print("\nTo stop the backend, close its console window or run:")
print(f"  taskkill /F /PID {server_process.pid}")
print("=" * 60)
