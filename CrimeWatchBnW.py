#!/usr/bin/env python3
# CrimeWatchBnW.py - Crime Detection in Video Footage
# A terminal-based application for crime detection in video footage

import sys
import os

# Check Python version
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required.")
    sys.exit(1)

# Check for required dependencies and provide installation guidance
try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required but not installed.")
    print("Please install it using: pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: Matplotlib is required but not installed.")
    print("Please install it using: pip install matplotlib")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: Pandas is required but not installed.")
    print("Please install it using: pip install pandas")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required but not installed.")
    print("Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is required but not installed.")
    print("Please install it using: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Error: Scikit-learn is required but not installed.")
    print("Please install it using: pip install scikit-learn")
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("Error: Imageio is required but not installed.")
    print("Please install it using: pip install imageio")
    sys.exit(1)

# Now that dependencies are checked, we can import everything else we need
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Input, Conv3D, MaxPooling3D
from tensorflow.keras.layers import GlobalAveragePooling3D, Bidirectional
from IPython.display import Image, display

# Global variables
DATA_DIR = 'dataset'
CRIME_DIR = os.path.join(DATA_DIR, 'crime')
SAFE_DIR = os.path.join(DATA_DIR, 'safe')
MODELS_DIR = 'models'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'crime'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'safe'), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Utility Functions
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a formatted header."""
    clear_screen()
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print()

def pause():
    """Pause execution until user presses Enter."""
    input("\nPress Enter to continue...")

# Video Analysis Functions
def get_video_info(video_path):
    """Get basic information about a video file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return frame_count, frame_width, frame_height

def analyze_dataset():
    """Analyze and display statistics about the dataset."""
    print_header("Dataset Analysis")
    
    # Check if dataset directories exist and contain videos
    if not os.path.exists(CRIME_DIR) or not os.path.exists(SAFE_DIR):
        print("Error: Dataset directories not found.")
        return
    
    crime_videos = [f for f in os.listdir(CRIME_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    safe_videos = [f for f in os.listdir(SAFE_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not crime_videos:
        print("No crime videos found in the dataset.")
    if not safe_videos:
        print("No safe videos found in the dataset.")
    if not crime_videos and not safe_videos:
        return
    
    print(f"Crime videos: {len(crime_videos)}")
    print(f"Safe videos: {len(safe_videos)}")
    print("\nAnalyzing video statistics...")
    
    crime_frame_counts = []
    crime_frame_widths = []
    crime_frame_heights = []
    
    safe_frame_counts = []
    safe_frame_widths = []
    safe_frame_heights = []
    
    # Analyze crime videos
    for video_file in crime_videos:
        video_path = os.path.join(CRIME_DIR, video_file)
        info = get_video_info(video_path)
        if info:
            crime_frame_counts.append(info[0])
            crime_frame_widths.append(info[1])
            crime_frame_heights.append(info[2])
    
    # Analyze safe videos
    for video_file in safe_videos:
        video_path = os.path.join(SAFE_DIR, video_file)
        info = get_video_info(video_path)
        if info:
            safe_frame_counts.append(info[0])
            safe_frame_widths.append(info[1])
            safe_frame_heights.append(info[2])
    
    # Print summary statistics
    if crime_frame_counts:
        print("\nCrime videos statistics:")
        print(f"Average frame count: {np.mean(crime_frame_counts):.0f}")
        print(f"Average frame width: {np.mean(crime_frame_widths):.0f}")
        print(f"Average frame height: {np.mean(crime_frame_heights):.0f}")
    
    if safe_frame_counts:
        print("\nSafe videos statistics:")
        print(f"Average frame count: {np.mean(safe_frame_counts):.0f}")
        print(f"Average frame width: {np.mean(safe_frame_widths):.0f}")
        print(f"Average frame height: {np.mean(safe_frame_heights):.0f}")
    
    # Ask if user wants to visualize the data
    choice = input("\nDo you want to see histograms of the data? (y/n): ")
    if choice.lower() == 'y':
        if crime_frame_counts:
            plot_histogram(crime_frame_counts, 'Frame Counts (CRIME)', 'Frame Count', 'Frequency')
        if crime_frame_widths:
            plot_histogram(crime_frame_widths, 'Frame Widths (CRIME)', 'Frame Width', 'Frequency')
        if crime_frame_heights:
            plot_histogram(crime_frame_heights, 'Frame Heights (CRIME)', 'Frame Height', 'Frequency')
        
        if safe_frame_counts:
            plot_histogram(safe_frame_counts, 'Frame Counts (SAFE)', 'Frame Count', 'Frequency')
        if safe_frame_widths:
            plot_histogram(safe_frame_widths, 'Frame Widths (SAFE)', 'Frame Width', 'Frequency')
        if safe_frame_heights:
            plot_histogram(safe_frame_heights, 'Frame Heights (SAFE)', 'Frame Height', 'Frequency')
    
    pause()

def plot_histogram(data, title, xlabel, ylabel):
    """Plot a histogram of the given data."""
    plt.figure()
    counts, bins, bars = plt.hist(data, bins=20, alpha=0.7, edgecolor='black')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, 
                 f'{int(count)}', ha='center', va='bottom')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_video():
    """Display a video from the dataset."""
    print_header("Video Visualization")
    
    # Check if dataset directories exist and contain videos
    if not os.path.exists(CRIME_DIR) or not os.path.exists(SAFE_DIR):
        print("Error: Dataset directories not found.")
        return
    
    crime_videos = [f for f in os.listdir(CRIME_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    safe_videos = [f for f in os.listdir(SAFE_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not crime_videos and not safe_videos:
        print("No videos found in the dataset.")
        return
    
    print("Choose a video type:")
    print("1. Crime Video")
    print("2. Safe Video")
    
    choice = input("\nEnter your choice (1-2): ")
    
    if choice == '1' and crime_videos:
        video_list = crime_videos
        video_dir = CRIME_DIR
        print("\nCrime Videos:")
    elif choice == '2' and safe_videos:
        video_list = safe_videos
        video_dir = SAFE_DIR
        print("\nSafe Videos:")
    else:
        print("Invalid choice or no videos available.")
        return
    
    for i, video in enumerate(video_list, 1):
        print(f"{i}. {video}")
    
    video_choice = input("\nEnter the number of the video to visualize (or 0 to cancel): ")
    
    try:
        video_idx = int(video_choice) - 1
        if video_idx < 0:
            return
        if video_idx >= len(video_list):
            print("Invalid video number.")
            return
        
        video_path = os.path.join(video_dir, video_list[video_idx])
        process_video_to_gif(video_path, frame_count=100, new_width=320, new_height=240)
    except ValueError:
        print("Please enter a valid number.")
    
    pause()

def process_video_to_gif(video_path, frame_count, new_width, new_height):
    """Convert a video to GIF and display it."""
    print(f"Processing video: {video_path}")
    print("This may take a moment...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        # Convert BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frames.append(resized_frame)
    
    cap.release()
    
    if len(frames) > frame_count:
        frames = frames[:frame_count]
    elif len(frames) < frame_count:
        last_frame = frames[-1] if frames else np.zeros((new_height, new_width, 3), dtype=np.uint8)
        frames += [last_frame] * (frame_count - len(frames))
    
    gif_path = 'output.gif'
    imageio.mimsave(gif_path, frames, fps=5)
    
    print(f"GIF saved to {gif_path}")
    print("Please open the GIF file to view it.")

# Model Functions
def extract_frames(video_path, frame_count, img_width, img_height):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_width, img_height))
        frames.append(frame)

    cap.release()

    if len(frames) > frame_count:
        frames = frames[:frame_count]
    elif len(frames) < frame_count:
        last_frame = frames[-1] if frames else np.zeros((img_height, img_width), dtype=np.uint8)
        frames += [last_frame] * (frame_count - len(frames))

    return np.array(frames).reshape(-1, img_height, img_width, 1)

def extract_and_fuse_frames(video_path, frame_count, img_width, img_height):
    """Extract frames from a video file and fuse them."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_width, img_height))
        frames.append(frame)

    cap.release()

    if len(frames) > frame_count:
        frames = frames[:frame_count]
    elif len(frames) < frame_count:
        last_frame = frames[-1] if frames else np.zeros((img_height, img_width), dtype=np.uint8)
        frames += [last_frame] * (frame_count - len(frames))

    frames = np.array(frames).reshape(-1, img_height, img_width, 1)
    fused_frames = np.mean(frames, axis=0).astype(np.uint8)

    return fused_frames

def load_data(n, frame_count, img_width, img_height, model_type='single_frame'):
    """Load data for model training."""
    X, y = [], []
    
    for category, folder in zip([0, 1], [SAFE_DIR, CRIME_DIR]):
        dirs = os.listdir(folder)[n:n+10]
        for video_file in dirs:
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_path = os.path.join(folder, video_file)
            
            if model_type == 'early_fusion':
                frames = extract_and_fuse_frames(video_path, frame_count, img_width, img_height)
                X.append(frames)
                y.append(category)
            else:  # single_frame, slow_fusion, or late_fusion
                frames = extract_frames(video_path, frame_count, img_width, img_height)
                if len(frames) == frame_count:
                    X.append(frames)
                    y.append(category)
    
    return np.array(X), np.array(y)

def load_all_data(frame_count, img_width, img_height, model_type='single_frame'):
    """Load all data for model evaluation."""
    X, y = [], []
    
    for category, folder in zip([0, 1], [SAFE_DIR, CRIME_DIR]):
        for video_file in os.listdir(folder):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_path = os.path.join(folder, video_file)
            
            if model_type == 'early_fusion':
                frames = extract_and_fuse_frames(video_path, frame_count, img_width, img_height)
                X.append(frames)
                y.append(category)
            else:  # single_frame, slow_fusion, or late_fusion
                frames = extract_frames(video_path, frame_count, img_width, img_height)
                if len(frames) == frame_count:
                    X.append(frames)
                    y.append(category)
    
    return np.array(X), np.array(y)

# Model creation functions
def create_single_frame_model(frame_count, img_height, img_width):
    """Create a single frame classification model."""
    model = Sequential()

    # Adjust for 3D Convolution
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(frame_count, img_height, img_width, 1)))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))

    # Flatten or use global pooling
    model.add(GlobalAveragePooling3D())

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_early_fusion_model(img_height, img_width):
    """Create an early fusion model."""
    model = Sequential()

    # Enhanced CNN layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # More complex Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    return model

def create_slow_fusion_model(frame_count, img_height, img_width):
    """Create a slow fusion model."""
    model = Sequential()

    # Slow fusion: start with TimeDistributed layers to process each frame, then gradually merge
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(frame_count, img_height, img_width, 1)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))

    # Slowly fuse temporal information with LSTM
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(LSTM(64))

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    return model

def create_late_fusion_model(frame_count, img_height, img_width):
    """Create a late fusion model."""
    # CNN for frame analysis
    cnn = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten()
    ])

    # TimeDistributed layer to process each frame
    input_layer = Input(shape=(frame_count, img_height, img_width, 1))
    time_dist = TimeDistributed(cnn)(input_layer)

    # LSTM for understanding temporal dynamics
    lstm_out = LSTM(64)(time_dist)

    # Dense layers for final decision making
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1, activation='sigmoid')(dense_out)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    return model

def train_model():
    """Train a new model or continue training an existing model."""
    print_header("Model Training")
    
    print("Select model type to train:")
    print("1. Single Frame Video Classification")
    print("2. Early Fusion Video Classification")
    print("3. Slow Fusion Video Classification")
    print("4. Late Fusion Video Classification")
    print("0. Return to Main Menu")
    
    choice = input("\nEnter your choice (0-4): ")
    
    if choice == '0':
        return
    
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice.")
        pause()
        return
    
    # Set parameters based on model type
    if choice == '1':  # Single Frame
        model_type = 'single_frame'
        frame_count = 500
        img_width, img_height = 60, 20
        model_file = os.path.join(MODELS_DIR, 'single_frame.keras')
        create_model_fn = create_single_frame_model
    elif choice == '2':  # Early Fusion
        model_type = 'early_fusion'
        frame_count = 1000
        img_width, img_height = 80, 50
        model_file = os.path.join(MODELS_DIR, 'early_fusion.keras')
        create_model_fn = create_early_fusion_model
    elif choice == '3':  # Slow Fusion
        model_type = 'slow_fusion'
        frame_count = 500
        img_width, img_height = 60, 20
        model_file = os.path.join(MODELS_DIR, 'slow_fusion.keras')
        create_model_fn = create_slow_fusion_model
    elif choice == '4':  # Late Fusion
        model_type = 'late_fusion'
        frame_count = 500
        img_width, img_height = 60, 20
        model_file = os.path.join(MODELS_DIR, 'late_fusion.keras')
        create_model_fn = create_late_fusion_model
    
    # Check if model already exists
    if os.path.exists(model_file):
        print(f"\nModel file {model_file} already exists.")
        print("1. Continue training existing model")
        print("2. Train new model (will overwrite existing model)")
        print("0. Cancel")
        
        subchoice = input("\nEnter your choice (0-2): ")
        
        if subchoice == '0':
            return
        
        if subchoice == '1':
            try:
                if model_type == 'single_frame':
                    model = load_model(model_file)
                else:
                    # For models that use legacy Adam optimizer
                    custom_objects = {'Adam': tf.keras.optimizers.legacy.Adam}
                    model = load_model(model_file, custom_objects=custom_objects)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                pause()
                return
        elif subchoice == '2':
            if model_type == 'early_fusion':
                model = create_model_fn(img_height, img_width)
            else:
                model = create_model_fn(frame_count, img_height, img_width)
        else:
            print("Invalid choice.")
            pause()
            return
    else:
        if model_type == 'early_fusion':
            model = create_model_fn(img_height, img_width)
        else:
            model = create_model_fn(frame_count, img_height, img_width)
    
    # Ask for training parameters
    try:
        epochs = int(input("\nEnter number of epochs (default: 10): ") or "10")
        batch_size = int(input("Enter batch size (default: 32): ") or "32")
    except ValueError:
        print("Invalid input. Using default values.")
        epochs = 10
        batch_size = 32
    
    # Train the model
    print("\nLoading training data...")
    try:
        X, y = load_data(0, frame_count, img_width, img_height, model_type)
        if len(X) == 0:
            print("No suitable data found. Please check your dataset.")
            pause()
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nTraining model with {len(X_train)} samples...")
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                          epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save the model
        model.save(model_file)
        print(f"\nModel saved to {model_file}")
        
        # Show training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during training: {e}")
    
    pause()

def test_model():
    """Test a trained model on a video."""
    print_header("Model Testing")
    
    # Check if models exist
    model_files = [
        os.path.join(MODELS_DIR, 'single_frame.keras'),
        os.path.join(MODELS_DIR, 'early_fusion.keras'),
        os.path.join(MODELS_DIR, 'slow_fusion.keras'),
        os.path.join(MODELS_DIR, 'late_fusion.keras')
    ]
    
    available_models = [os.path.exists(model) for model in model_files]
    
    if not any(available_models):
        print("No trained models found. Please train a model first.")
        pause()
        return
    
    print("Select model to test:")
    
    model_names = ["Single Frame", "Early Fusion", "Slow Fusion", "Late Fusion"]
    model_types = ["single_frame", "early_fusion", "slow_fusion", "late_fusion"]
    
    for i, (name, available) in enumerate(zip(model_names, available_models), 1):
        if available:
            print(f"{i}. {name}")
        else:
            print(f"{i}. {name} (not available)")
    
    print("0. Return to Main Menu")
    
    choice = input("\nEnter your choice (0-4): ")
    
    if choice == '0':
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_names):
            print("Invalid choice.")
            pause()
            return
        
        if not available_models[idx]:
            print(f"Model {model_names[idx]} is not available. Please train it first.")
            pause()
            return
        
        model_type = model_types[idx]
        model_file = model_files[idx]
        
        # Set parameters based on model type
        if model_type == 'single_frame':
            frame_count = 500
            img_width, img_height = 60, 20
        elif model_type == 'early_fusion':
            frame_count = 1000
            img_width, img_height = 80, 50
        else:  # slow_fusion or late_fusion
            frame_count = 500
            img_width, img_height = 60, 20
        
        # Choose a video to test
        print("\nChoose a video to test:")
        print("1. Use a video from the crime dataset")
        print("2. Use a video from the safe dataset")
        print("3. Use a custom video path")
        
        video_choice = input("\nEnter your choice (1-3): ")
        
        if video_choice == '1':
            videos = [f for f in os.listdir(CRIME_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if not videos:
                print("No crime videos found.")
                pause()
                return
            
            print("\nAvailable crime videos:")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            
            video_idx = int(input("\nEnter video number: ")) - 1
            if video_idx < 0 or video_idx >= len(videos):
                print("Invalid video number.")
                pause()
                return
            
            video_path = os.path.join(CRIME_DIR, videos[video_idx])
            
        elif video_choice == '2':
            videos = [f for f in os.listdir(SAFE_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if not videos:
                print("No safe videos found.")
                pause()
                return
            
            print("\nAvailable safe videos:")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            
            video_idx = int(input("\nEnter video number: ")) - 1
            if video_idx < 0 or video_idx >= len(videos):
                print("Invalid video number.")
                pause()
                return
            
            video_path = os.path.join(SAFE_DIR, videos[video_idx])
            
        elif video_choice == '3':
            video_path = input("\nEnter the full path to the video file: ")
            if not os.path.exists(video_path):
                print("File not found.")
                pause()
                return
        else:
            print("Invalid choice.")
            pause()
            return
        
        # Load the model
        try:
            if model_type == 'single_frame':
                model = load_model(model_file)
            else:
                # For models that use legacy Adam optimizer
                custom_objects = {'Adam': tf.keras.optimizers.legacy.Adam}
                model = load_model(model_file, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            pause()
            return
        
        # Process the video
        print(f"\nProcessing video: {video_path}")
        try:
            if model_type == 'early_fusion':
                processed_video = extract_and_fuse_frames(video_path, frame_count, img_width, img_height)
                processed_video = np.expand_dims(processed_video, axis=0)
            else:
                processed_video = extract_frames(video_path, frame_count, img_width, img_height)
                processed_video = np.expand_dims(processed_video, axis=0)
            
            # Make prediction
            print("Making prediction...")
            prediction = model.predict(processed_video)
            predicted_class = 'Crime' if prediction[0][0] > 0.5 else 'Safe'
            confidence = prediction[0][0] if predicted_class == 'Crime' else 1 - prediction[0][0]
            
            print(f"\nResult: The video is predicted as: {predicted_class}")
            print(f"Confidence: {confidence*100:.2f}%")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
        
    except ValueError:
        print("Please enter a valid number.")
    
    pause()

def evaluate_models():
    """Evaluate all available models on the full dataset."""
    print_header("Model Evaluation")
    
    # Check if models exist
    model_files = [
        os.path.join(MODELS_DIR, 'single_frame.keras'),
        os.path.join(MODELS_DIR, 'early_fusion.keras'),
        os.path.join(MODELS_DIR, 'slow_fusion.keras'),
        os.path.join(MODELS_DIR, 'late_fusion.keras')
    ]
    
    model_names = ["Single Frame", "Early Fusion", "Slow Fusion", "Late Fusion"]
    model_types = ["single_frame", "early_fusion", "slow_fusion", "late_fusion"]
    
    available_models = [os.path.exists(model) for model in model_files]
    
    if not any(available_models):
        print("No trained models found. Please train a model first.")
        pause()
        return
    
    print("Available models for evaluation:")
    available_indices = []
    
    for i, (name, available) in enumerate(zip(model_names, available_models)):
        if available:
            print(f"{i+1}. {name}")
            available_indices.append(i)
    
    print("0. Return to Main Menu")
    
    choice = input("\nEnter your choice (0 or one of the available models): ")
    
    if choice == '0':
        return
    
    try:
        idx = int(choice) - 1
        if idx not in available_indices:
            print("Invalid choice or model not available.")
            pause()
            return
        
        model_type = model_types[idx]
        model_file = model_files[idx]
        
        # Set parameters based on model type
        if model_type == 'single_frame':
            frame_count = 500
            img_width, img_height = 60, 20
        elif model_type == 'early_fusion':
            frame_count = 1000
            img_width, img_height = 80, 50
        else:  # slow_fusion or late_fusion
            frame_count = 500
            img_width, img_height = 60, 20
        
        # Load the model
        try:
            if model_type == 'single_frame':
                model = load_model(model_file)
            else:
                # For models that use legacy Adam optimizer
                custom_objects = {'Adam': tf.keras.optimizers.legacy.Adam}
                model = load_model(model_file, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            pause()
            return
        
        # Evaluate on full dataset
        print(f"\nEvaluating {model_names[idx]} model on the full dataset...")
        print("This may take some time depending on the size of your dataset.")
        
        try:
            X, y = load_all_data(frame_count, img_width, img_height, model_type)
            
            if len(X) == 0:
                print("No suitable data found. Please check your dataset.")
                pause()
                return
            
            loss, accuracy = model.evaluate(X, y, verbose=1)
            print(f"\nResults for {model_names[idx]}:")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy*100:.2f}%")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
    except ValueError:
        print("Please enter a valid number.")
    
    pause()

# Main Menu Functions
def show_main_menu():
    """Display the main menu."""
    print_header("CrimeWatch - Crime Detection in Video Footage")
    
    print("Main Menu:")
    print("1. Analyze Dataset")
    print("2. Visualize Video")
    print("3. Train Model")
    print("4. Test Model")
    print("5. Evaluate Models on Full Dataset")
    print("6. Help / About")
    print("0. Exit")

def show_help():
    """Display help information."""
    print_header("Help / About")
    
    print("CrimeWatch - Crime Detection in Video Footage")
    print("Version 1.0")
    print("\nThis application uses machine learning to detect criminal activity in video footage.")
    print("\nDataset Structure:")
    print("- dataset/crime/ - Contains video files showing criminal activity")
    print("- dataset/safe/ - Contains video files without criminal activity")
    print("\nModels:")
    print("1. Single Frame Video Classification")
    print("   - Processes each frame individually with 3D convolutions")
    print("2. Early Fusion Video Classification")
    print("   - Fuses all frames into a single image before classification")
    print("3. Slow Fusion Video Classification")
    print("   - Gradually fuses information across frames")
    print("4. Late Fusion Video Classification")
    print("   - Processes each frame separately, then fuses the results")
    print("\nUsage:")
    print("1. First, make sure you have videos in the dataset directories")
    print("2. Analyze the dataset to understand its characteristics")
    print("3. Train one or more models")
    print("4. Test the models on specific videos")
    print("5. Evaluate the models on the full dataset")
    
    pause()

def main():
    """Main program entry point."""
    while True:
        show_main_menu()
        
        choice = input("\nEnter your choice (0-6): ")
        
        if choice == '0':
            print("\nExiting CrimeWatch. Goodbye!")
            break
        elif choice == '1':
            analyze_dataset()
        elif choice == '2':
            visualize_video()
        elif choice == '3':
            train_model()
        elif choice == '4':
            test_model()
        elif choice == '5':
            evaluate_models()
        elif choice == '6':
            show_help()
        else:
            print("Invalid choice. Please try again.")
            pause()

if __name__ == "__main__":
    # Check for TensorFlow
    print("Checking TensorFlow...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPU is available for TensorFlow: {len(physical_devices)} device(s) found.")
        # Allow memory growth for the GPU
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except:
                print("Memory growth cannot be enabled")
    else:
        print("GPU is not available. Using CPU for TensorFlow (this may be slower).")
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'crime'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'safe'), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Start the main program
    main()



