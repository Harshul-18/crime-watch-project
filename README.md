# CrimeWatch

A project for crime detection in video footage using machine learning.

Article: [A Comprehensive Study to Predicting Unethical: Activity in Videos Through Deep Learning Techniques](https://link.springer.com/chapter/10.1007/978-981-97-9132-3_11)

## Features

- Single Frame Video Classification
- Early Fusion Video Classification
- Slow Fusion Video Classification
- Late Fusion Video Classification
- Dataset analysis and visualization
- Training and testing of models
- Evaluation of models on full dataset

## Installation

1. Make sure you have Python 3.6 or higher installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application in terminal:

```bash
python CrimeWatchBnW.py
```
> [NOTE] - Visualization may not work as required, since it is a terminal based application.

2. Run the jupyter notebook `CrimeWatchBnW.ipynb`.

## Dataset Structure

Place your video files in the appropriate directories:

- `dataset/crime/` - Contains video files showing criminal activity
- `dataset/safe/` - Contains video files without criminal activity

The application will create these directories if they don't exist.

## Models

The application uses the following models:

1. **Single Frame Video Classification**
   - Processes each frame individually with 3D convolutions

2. **Early Fusion Video Classification**
   - Fuses all frames into a single image before classification

3. **Slow Fusion Video Classification**
   - Gradually fuses information across frames

4. **Late Fusion Video Classification**
   - Processes each frame separately, then fuses the results

Trained models are saved in the `models/` directory. 

## Copyright Notice

Copyright © 2023. All Rights Reserved.

This project is provided for portfolio and demonstration purposes only. 
