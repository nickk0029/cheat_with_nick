# cheat_with_nick
An advanced, low-latency AI-powered desktop assistant designed to capture on-screen problem statements and deliver instant, context-aware solutions. The system bridges screen capture utilities with large language models to process complex technical queries, algorithms, and theoretical questions seamlessly in the background.
# AI Overlay Assistant

A desktop overlay that answers questions from text, voice or a screenshot, built in Python.

## Features
- Type a question and get a streamed answer
- Voice input with local speech-to-text (faster-whisper)
- Screenshot button that reads questions off your screen
- Always-on-top window with a transparency slider
- Global hotkeys: Ctrl+Shift+H (show/hide), Ctrl+Shift+S (screenshot)

## Setup
1. Install Python 3.10+
2. pip install -r requirements.txt
3. Get a free key at aistudio.google.com/apikey
4. Create a file named .env containing: GEMINI_API_KEY=your_key
5. Run: python ai_overlay.py

## Progress log
- Day 1: text, voice and screenshot working
