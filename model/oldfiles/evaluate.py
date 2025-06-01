#!/usr/bin/env python3
"""
Main entry point for kanji evaluation
Run from kanjinin/model directory
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))

from inference.evaluate_student_kanji import main

if __name__ == "__main__":
    main()
