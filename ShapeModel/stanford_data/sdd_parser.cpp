// 1) parse annotatio file and create hashtable
// 2) parse corresponding video 
// - > for each frame, check if frame number exists in hastable, if yes then save image, if no then skip frame
// 3) have annotation txt files and img files saved in dataset folder in filesystem

//includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

#include "structs.h" 
#include "helpers.h"
#include "sdd_parser.h"


//parses annotation file and makes hashtable
static std::unordered_map<int, std::vector<Annotation>> parse_annotations(const std::string& text_path) {
    // Open the file
    std::ifstream file(text_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return {};  // Return empty map on error
    }

    // Create a hashtable (unordered_map) to store annotations
    std::unordered_map<int, std::vector<Annotation>> hashtable;

    std::string line;
    
    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        
        // Parse the line
        int id_track, x1, y1, x2, y2, frame, lost, dummy1, dummy2;
        std::string label;
        
        // Extract each value
        iss >> id_track >> x1 >> y1 >> x2 >> y2 >> frame >> lost >> dummy1 >> dummy2;
        
        // Extract "Pedestrian" (or whatever string is in quotes)
        // This handles the quoted string at the end
        std::string quote_part;
        getline(iss, quote_part, '"');  // Skip until first quote
        getline(iss, label, '"');  // Get content between quotes
        
        // Get the numeric code for the label
        int labelCode = getLabelCode(label);
        
        // Create formatted string: "label_number x1 y1 x2 y2"
        std::ostringstream formatted_data;
        formatted_data << labelCode << " " << x1 << " " << y1 << " " << x2 << " " << y2;
        
        // Create an annotation with the formatted data
        Annotation annotation;
        annotation.data = formatted_data.str();
        
        // Add to the hashtable using frame as the key
        // Only add if the track is not lost (eighth number is 0)
        if (lost == 0) {
            hashtable[frame].push_back(annotation);
        }
    }
    
    // Close the file
    file.close();
    
    return hashtable;
}

// Function to save annotation file for a specific frame
static void save_annotation_file(int frameNumber, const std::vector<Annotation>& frameAnnotations, 
                         const std::string& baseFilename, const std::string& output_dir) {
    // Create output filename: baseFilename_frameNumber.txt
    std::ostringstream outputFilenameStream;
    outputFilenameStream << output_dir << "/" << baseFilename << "_" << frameNumber << ".txt";
    std::string outputFilename = outputFilenameStream.str();
    
    // Open output file
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilename << std::endl;
        return;  // Return on error
    }
    
    // Write annotations to file (one per line)
    for (const auto& annotation : frameAnnotations) {
        outputFile << annotation.data << std::endl;
    }
    
    // Close output file
    outputFile.close();
    
    std::cout << "Created annotation file: " << outputFilename << " with " 
              << frameAnnotations.size() << " annotation(s)" << std::endl;
}

// Function to save a video frame as an image
static void save_frame_image(const cv::Mat& frame, int frameNumber, 
                     const std::string& baseFilename, const std::string& output_dir) {
    std::string imagePath = output_dir + "/" + baseFilename + "_" + std::to_string(frameNumber) + ".jpg";
    bool success = cv::imwrite(imagePath, frame);
    
    if (success) {
        std::cout << "Saved image: " << imagePath << std::endl;
    } else {
        std::cerr << "Error saving image: " << imagePath << std::endl;
    }
}

// Main function to process video and save annotations and images for frames in hashtable
void process_video(const std::string& video_path, const std::string& text_path,
                  const std::vector<std::string>& output_dirs, float split, int frame_interval) {
    // Validate output_dirs size
    if (output_dirs.size() != 4) {
        std::cerr << "Error: output_dirs must contain exactly 4 paths (train_img, train_label, val_img, val_label)" << std::endl;
        return;
    }
    
    // Extract output directories
    const std::string& train_img_dir = output_dirs[0];
    const std::string& train_label_dir = output_dirs[1];
    const std::string& val_img_dir = output_dirs[2];
    const std::string& val_label_dir = output_dirs[3];
    
    // Parse annotations to get the hashtable
    auto hashtable = parse_annotations(text_path);
    
    // Get base filename without extension for both annotation files and images
    //std::string baseFilename = getBaseFilename(text_path);
    std::string videoBaseName = getBaseFilename(video_path);
    
    // Open the video file
    cv::VideoCapture video(video_path);
    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video file " << video_path << std::endl;
        return;
    }
    
    int frame_count = 0;
    int processed_count = 0;
    int train_count = 0;
    int val_count = 0;
    
    // Random number generator for consistent split
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    
    // Process video frame by frame
    while (true) {
        cv::Mat frame;
        bool ret = video.read(frame);
        
        if (!ret) {
            break;  // End of video
        }
        
        // Check if the current frame number exists in the hashtable
        auto it = hashtable.find(frame_count);
        if (it != hashtable.end() && frame_count % frame_interval == 0) {
            // Determine if this frame goes to training or validation set
            bool is_train = (dis(gen) < split);
            
            // Set appropriate directories based on train/val decision
            const std::string& img_dir = is_train ? train_img_dir : val_img_dir;
            const std::string& label_dir = is_train ? train_label_dir : val_label_dir;
            
            // Save annotation file in appropriate directory
            save_annotation_file(frame_count, it->second, videoBaseName, label_dir);
            
            // Save frame image in appropriate directory
            save_frame_image(frame, frame_count, videoBaseName, img_dir);
            
            // Update counts
            processed_count++;
            is_train ? train_count++ : val_count++;
        }
        
        frame_count++;
    }
    
    video.release();
    std::cout << "Processing complete! Processed " << processed_count << " frames total." << std::endl;
    std::cout << "Train set: " << train_count << " frames (" 
              << (processed_count > 0 ? 100.0f * train_count / processed_count : 0) << "%)" << std::endl;
    std::cout << "Validation set: " << val_count << " frames (" 
              << (processed_count > 0 ? 100.0f * val_count / processed_count : 0) << "%)" << std::endl;
}


