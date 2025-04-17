#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sys/stat.h>

// Function to create directories on macOS
void create_directories(const std::string& path) {
    std::string command = "mkdir -p \"" + path + "\"";
    system(command.c_str());
}

bool directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

void grab_screenshots(const std::string& video_path, const std::string& output_dir, int frame_interval) {
    // Create output directory if it doesn't exist
    if (!directory_exists(output_dir)) {
        create_directories(output_dir);
    }

    // Open the video file
    cv::VideoCapture video(video_path);

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video file " << video_path << std::endl;
        return;
    }

    int frame_count = 0;
    int saved_count = 0;

    while (true) {
        cv::Mat frame;
        bool ret = video.read(frame);

        if (!ret) {
            break;
        }

        if (frame_count % frame_interval == 0) {
            std::string screenshot_path = output_dir + "/0040frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(screenshot_path, frame);
            saved_count++;
            std::cout << "Saved screenshot: " << screenshot_path << std::endl;
        }

        frame_count++;
    }

    video.release();
    std::cout << "Done! Saved " << saved_count << " screenshots." << std::endl;
}

int main() {
    // macOS style paths (using forward slashes)
    std::string input_video = "bookstore_video_0.mov";
    std::string output_folder = "test";
    
    grab_screenshots(input_video, output_folder, 100);
    return 0;
}