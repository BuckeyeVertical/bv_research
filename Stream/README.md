# USB Camera Stream Setup

This directory contains a USB camera streaming application that can capture live video, record footage, and save frames from any connected USB camera.

## Features

- **Live Video Stream**: Real-time display of USB camera feed
- **Video Recording**: Start/stop recording with keyboard controls
- **Frame Capture**: Save individual frames as images
- **FPS Monitoring**: Real-time frames-per-second display
- **Configurable Settings**: Customizable resolution, frame rate, and camera selection
- **Interactive Controls**: Keyboard shortcuts for all functions

## Prerequisites

- Python 3.8 or higher
- USB camera connected to your system
- uv package manager installed

### Installing uv

If you don't have uv installed, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup Instructions

### 1. Navigate to Stream Directory

```bash
cd Stream
```

### 2. Create Virtual Environment and Install Dependencies

Using uv, create a virtual environment and install all required packages:

```bash
# Create virtual environment and install dependencies
uv sync

# Or if you prefer to create environment manually:
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Verify Camera Connection

Check that your USB camera is detected:

```bash
# List video devices
ls /dev/video*

# Get detailed camera info
v4l2-ctl --list-devices
```

## Usage

### Basic Usage

Run the camera stream with default settings:

```bash
uv run python stream.py
```

### Advanced Usage

#### Specify Camera and Resolution

```bash
# Use camera 1 with HD resolution
uv run python stream.py --camera 1 --width 1280 --height 720 --fps 30
```

#### Enable Automatic Frame Saving

```bash
# Save every 60th frame
uv run python stream.py --save-frames --frame-interval 60
```

#### Hide FPS Counter

```bash
uv run python stream.py --no-fps
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--camera` | `-c` | Camera device ID | 0 |
| `--width` | `-w` | Frame width in pixels | 640 |
| `--height` | `-h` | Frame height in pixels | 480 |
| `--fps` | `-f` | Frames per second | 30 |
| `--no-fps` | | Hide FPS counter | False |
| `--save-frames` | | Save frames periodically | False |
| `--frame-interval` | | Save every Nth frame | 30 |

## Interactive Controls

While the stream is running, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit application |
| `r` | Start/Stop video recording |
| `s` | Save current frame as image |
| `f` | Toggle FPS display on/off |

## Output Files

The application creates several types of output files:

- **Video recordings**: `recording_YYYYMMDD_HHMMSS.mp4`
- **Saved frames**: `saved_frame_YYYYMMDD_HHMMSS_mmm.jpg`
- **Auto-saved frames**: `frame_YYYYMMDD_HHMMSS_mmm.jpg`

All files are saved in the current working directory.

## Troubleshooting

### Camera Not Detected

1. **Check camera connection**: Ensure USB camera is properly connected
2. **Check permissions**: You may need camera permissions:
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in for changes to take effect
   ```
3. **Try different camera ID**: Use `--camera 1` or `--camera 2` if default doesn't work

### Low Performance Issues

1. **Reduce resolution**: Use lower width/height values
   ```bash
   uv run python stream.py --width 320 --height 240
   ```

2. **Lower frame rate**: Reduce FPS setting
   ```bash
   uv run python stream.py --fps 15
   ```

3. **Close other applications**: Free up system resources

### OpenCV Installation Issues

If you encounter OpenCV-related errors:

```bash
# Reinstall OpenCV
uv pip uninstall opencv-python
uv pip install opencv-python

# For headless systems without display, try headless version
uv pip install opencv-python-headless
```

## Development

### Project Structure

```
Stream/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── stream.py          # Main streaming application
└── .venv/             # Virtual environment (created by uv)
```

### Adding Features

The `USBCameraStreamer` class is modular and can be extended with additional features:

- Object detection integration
- Motion detection
- Multiple camera support
- Network streaming capabilities

## System Requirements

- **Operating System**: Linux
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended for HD streaming)
- **USB**: USB 2.0 or higher port for camera connection

## License

This project follows the same license as the parent repository.