Overview
ORBIS (Offline Real-time Blindness Intelligence System) is an AI-powered smart assistive system designed to improve independence and environmental awareness for visually impaired individuals. By combining Computer Vision, Edge AI, Depth Estimation, and Natural Language Scene Understanding, ORBIS provides real-time audio descriptions of the user's surroundings without requiring continuous internet connectivity.
The system captures live video through a camera, analyzes the environment using optimized AI models, and delivers spoken feedback through a bone-conduction audio interface. This allows users to navigate safely while remaining aware of ambient sounds.

Problem Statement
Millions of visually impaired individuals face challenges in navigating unfamiliar environments, identifying obstacles, recognizing objects, and understanding dynamic surroundings. Existing assistive technologies often provide limited functionality, depend heavily on internet connectivity, or are prohibitively expensive.
ORBIS addresses these challenges through an affordable, wearable, and offline-capable AI assistant that delivers contextual environmental awareness in real time.

Objectives
Detect and identify objects in the user's surroundings.
Estimate object distance and spatial relationships.
Generate contextual scene descriptions.
Provide real-time audio feedback.
Operate efficiently on edge devices.
Function with minimal or no internet connectivity.
Maintain user privacy through on-device processing.

Key Features
Real-Time Object Detection
Detects and classifies objects in the environment using YOLO.
Depth Awareness
Estimates object distance using MiDaS depth estimation.
Scene Understanding
Provides natural language descriptions of complex environments using vision-language models.
Offline Processing
Runs entirely on-device, reducing latency and improving privacy.
Audio Guidance
Converts scene information into spoken instructions for the user.
Edge AI Optimization
Utilizes model optimization techniques to achieve real-time performance on embedded hardware.

Hardware Components

Raspberry Pi 5 / Edge AI Device	Main processing unit
Camera Module	Captures live video
Bone Conduction Speaker	Audio feedback
USB Microphone	Voice commands
Power Bank/Battery	Portable power source
Smart Glasses Frame	Wearable housing
Software Stack
Computer Vision
OpenCV
NumPy
Deep Learning
PyTorch
Transformers

AI Models
YOLOv8 (Object Detection)
MiDaS (Depth Estimation)
BLIP / LLaVA (Scene Description)
Audio Processing
pyttsx3
SpeechRecognition
Operating System
Raspberry Pi OS
macOS (Development)

Project Pipeline
Stage 1: Visual Capture
The camera continuously captures frames from the environment.
Stage 2: Image Processing
Frames undergo preprocessing, enhancement, and filtering.
Stage 3: Object Detection
YOLO identifies objects and their locations.
Stage 4: Depth Estimation
MiDaS calculates relative object distances.
Stage 5: Scene Understanding
A multimodal AI model generates contextual descriptions.
Stage 6: Audio Feedback
Relevant information is converted into speech and delivered to the user.

Expected Outcomes
Enhanced situational awareness.
Improved navigation safety.
Increased independence for visually impaired users.
Reduced dependence on external assistance.
Affordable and scalable assistive technology.

Future Enhancements
Face Recognition
OCR-based Text Reading
GPS Navigation Assistance
Emergency Alert System
Multilingual Audio Feedback
Gesture-Based Controls
Edge TPU / NPU Acceleration
Custom Fine-Tuned Vision-Language Models

Project Team
Vishal M
Janani KS
Mohamed Anas
Maryann Joseph Bincy

Project Name: ORBIS
Developed By:
Department of Electronics and Communication Engineering
Institution:
SRM Institute of Science and Technology
Academic Year: 2026
License
This project is developed for educational, research, and social-impact purposes. Any deployment in real-world assistive environments should undergo extensive testing and validation before use.
