#!/usr/bin/env python3
"""Parse GCP log JSON and extract meaningful errors."""
import json
import sys

LOG_FILE = "downloaded-logs-20260208-091159.json"

NOISE = [
    "pythonjsonlogger", "Running pip as", "new release of pip",
    "pip install --upgrade pip", "f2py", "--no-warn-script",
    "ydata-profiling", "bigframes", "pandas-gbq", "ibis-framework",
    "numba 0.58", "dependency resolver does not",
]

with open(LOG_FILE) as f:
    logs = json.load(f)

print(f"Total entries: {len(logs)}\n")

# Show ALL non-empty INFO entries to see progress
for i, entry in enumerate(logs):
    sev = entry.get("severity", "")
    text = entry.get("textPayload", "")
    jpay = entry.get("jsonPayload", {})
    msg = jpay.get("message", "") if jpay else text
    if not msg.strip():
        continue
    if sev == "INFO":
        print(f"[{i}] INFO: {msg[:250]}")

print("\n\n=== ERRORS ===\n")

seen = set()
for i, entry in enumerate(logs):
    sev = entry.get("severity", "")
    text = entry.get("textPayload", "")
    jpay = entry.get("jsonPayload", {})
    msg = jpay.get("message", "") if jpay else text

    if not msg.strip():
        continue
    if any(n in msg for n in NOISE):
        continue

    is_error = (
        sev == "ERROR"
        or "Traceback" in msg
        or "Error" in msg
        or "Exception" in msg
        or "FAILED" in msg
    )
    if not is_error:
        continue

    short = msg[:200]
    if short in seen:
        continue
    seen.add(short)

    print(f"--- [{i}] severity={sev} ---")
    print(msg[:2000])
    print()
