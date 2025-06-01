#!/bin/bash
# Run test from the correct directory
cd "$(dirname "$0")"
cd inference
python3 test_system.py
