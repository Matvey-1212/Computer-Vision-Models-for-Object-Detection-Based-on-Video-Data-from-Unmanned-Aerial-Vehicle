# Drone detection for search and rescue operations in sparsely populated area in Russia
repo and readME in progress...


This repository contains an enhanced version of the RetinaNet model with several improvements designed for detecting lost persons in high-resolution images (8000x6000 pixels). The primary goal of this project is to develop a reliable and accurate detection system that can operate efficiently on low-power devices such as UAV onboard systems or laptops.

## Overview

Detecting lost persons in sparsely populated areas using high-resolution images presents unique challenges. This project enhances the baseline RetinaNet model to improve detection accuracy and performance, especially on devices with limited computational power.

## Features

- Enhanced feature pyramid with concatenation between layers.
- Additional level in the feature pyramid for finer detection.
- Integration of inception layers in the feature pyramid to capture multi-scale features.
- Application of attention layers to improve focus on significant parts of the image.
- Improved detection accuracy surpassing state-of-the-art models like Faster R-CNN for this specific task.

## Model Enhancements

1. **Feature Pyramid Concatenation**:
   - Layers of the feature pyramid network (FPN) are concatenated to leverage multi-scale feature representation.

2. **Additional Feature Pyramid Level**:
   - An extra level is added to the FPN, enabling the model to detect smaller objects with higher precision.

3. **Inception Layers**:
   - Inception modules are integrated within the FPN, allowing the network to capture diverse and complex feature patterns.

4. **Attention Layers**:
   - Attention mechanisms are applied to prioritize significant features, improving the detection of lost individuals in challenging environments.

## Image Examples

<img src="https://github.com/Matvey-1212/Computer-Vision-Models-for-Object-Detection-Based-on-Video-Data-from-Unmanned-Aerial-Vehicle/blob/main/example_data/main_data/train/images/1_000285.JPG" alt="Example Image" width="600">

## Comparison Table

| Model              | mAP   | mFscore (0.5) | FPS  | Model Size (MB) |
|--------------------|-------|---------------|------|-----------------|
| Faster-RCNN        | 0.292 | 0.496         | 7.4  | 158.3           |
|
