# 🚧 AI Pothole Detection System

An AI-powered computer vision system that automatically detects potholes in road images and videos using **Roboflow Workflow and Python**.

The system processes road footage, identifies potholes, calculates detection confidence, and draws bounding boxes around detected potholes.

---

## 📌 Project Overview

Potholes are a major road-safety and infrastructure problem. Manual road inspection is time-consuming and difficult to scale.

This project uses **AI-based object detection** to automatically identify potholes from road images and driving videos.

### 🎯 Objective

Build a computer vision system that can:

- Detect potholes automatically
- Identify pothole locations using bounding boxes
- Calculate detection confidence
- Process road-driving videos
- Provide visual detection results
- Support future real-time camera deployment

---

## 🧠 How It Works

```text
Road Image / Video
        │
        ▼
   OpenCV Processing
        │
        ▼
 Extract Video Frames
        │
        ▼
   Roboflow Workflow
        │
        ▼
   AI Object Detection
        │
        ▼
  Pothole Detection
        │
        ▼
Bounding Box + Confidence
        │
        ▼
   Output Video
🔍 Detection Class

The current model focuses on a single object class:

pothole

Example detection:

POTHOLE 87.4%
┌─────────────────────┐
│                     │
│      POTHOLE        │
│                     │
└─────────────────────┘
🛠️ Technologies Used
Technology	Purpose
Python	Core programming
OpenCV	Video processing
Roboflow	AI model & workflow
Roboflow Inference SDK	API integration
Computer Vision	Pothole detection
Git & GitHub	Version control
📂 Project Structure
pothole-detection-ai/
│
├── pothole_video.py
├── README.md
├── .gitignore
│
└── pothole_test.mp4

Large video files and sensitive API keys should not be committed to GitHub.

⚙️ Installation
1. Clone the repository
git clone https://github.com/akshayh0/pothole-detection-ai.git
cd pothole-detection-ai
2. Install Python dependencies

Python 3.12 is recommended.

py -3.12 -m pip install inference-sdk opencv-python
🔑 Roboflow API Configuration

Create a Roboflow API key and keep it private.

Do not hard-code your API key in the source code.

Recommended approach:

import os

API_KEY = os.getenv("ROBOFLOW_API_KEY")

On Windows PowerShell:

$env:ROBOFLOW_API_KEY="YOUR_API_KEY"
▶️ Run the Project

Place your test video in the project directory:

pothole_test.mp4

Then run:

py -3.12 pothole_video.py

The program processes the video and generates:

pothole_detection_output.mp4
🎥 Video Detection

The system:

Opens the input road video
Extracts video frames
Sends selected frames to the Roboflow workflow
Detects potholes
Retrieves bounding-box coordinates
Calculates confidence
Draws detection boxes
Generates an output video
📊 Example Prediction

A Roboflow prediction contains information such as:

{
  "class": "pothole",
  "confidence": 0.59,
  "x": 306,
  "y": 158,
  "width": 100,
  "height": 14
}

The bounding-box coordinates are converted into:

(x1, y1)
      ┌──────────────────┐
      │     POTHOLE      │
      │                  │
      └──────────────────┘
                       (x2, y2)
