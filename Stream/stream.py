import cv2
import sys
import time
import argparse
from datetime import datetime

class USBCameraStreamer:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Initialize USB camera streamer.
        
        Args:
            camera_id: Camera device ID (usually 0 for first camera)
            width: Frame width
            height: Frame height  
            fps: Frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        
    def initialize_camera(self):
        """Initialize and configure the camera."""
        print(f"Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual properties (camera might not support requested values)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")
        
        return True
        
    def start_recording(self, output_filename=None):
        """Start recording video to file."""
        if self.is_recording:
            print("Already recording!")
            return
            
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"recording_{timestamp}.mp4"
            
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_writer = cv2.VideoWriter(
            output_filename, fourcc, self.fps, (actual_width, actual_height)
        )
        
        self.is_recording = True
        print(f"Started recording to {output_filename}")
        
    def stop_recording(self):
        """Stop recording video."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("Stopped recording")
        
    def stream(self, show_fps=True, save_frames=False, frame_save_interval=30):
        """
        Start the camera stream with display.
        
        Args:
            show_fps: Whether to display FPS counter
            save_frames: Whether to save frames periodically
            frame_save_interval: Save every Nth frame if save_frames=True
        """
        if not self.initialize_camera():
            return False
            
        print("\nCamera Stream Started!")
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  'r' - Start/Stop recording")
        print("  's' - Save current frame")
        print("  'f' - Toggle FPS display")
        print()
        
        frame_count = 0
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                    
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()
                
                # Add FPS text to frame
                if show_fps:
                    cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add recording indicator
                if self.is_recording:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (frame.shape[1] - 60, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Save frame to video file if recording
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Save individual frames if requested
                if save_frames and frame_count % frame_save_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved frame: {frame_filename}")
                
                # Display the frame
                cv2.imshow('USB Camera Stream', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Toggle recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('s'):  # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_filename = f"saved_frame_{timestamp}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved frame: {frame_filename}")
                elif key == ord('f'):  # Toggle FPS display
                    show_fps = not show_fps
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            self.cleanup()
            
        return True
        
    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("Camera stream ended")

def main():
    parser = argparse.ArgumentParser(description='USB Camera Streamer')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', '-w', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', '-h', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save frames periodically')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Save every Nth frame (default: 30)')
    
    args = parser.parse_args()
    
    # Create and start the camera streamer
    streamer = USBCameraStreamer(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    success = streamer.stream(
        show_fps=not args.no_fps,
        save_frames=args.save_frames,
        frame_save_interval=args.frame_interval
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
