# Skill: MediaPipe 0.10.32+ Migration Fix

## Description
Fix for MediaPipe API breaking changes in versions 0.10.32 and later. Replaces deprecated `mp.solutions` with the new Tasks API.

## Context
- When: MediaPipe import errors, `AttributeError: module 'mediapipe' has no attribute 'solutions'`
- Files: Any file using MediaPipe face detection/landmarks

## Fix

### Required model download
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task