// 1) take in annotation txt file and video mov file.
// 2) run annotation parser and initialize dataset, save 80% of annotation files to dataset/train/labels and 20% to dataset/valid/labels
// 3) run video parser and save 80% of image files to dataset/train/images and 20% to dataset/valid/images

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "structs.h"
#include "helpers.h"
#include "sdd_parser.h"
#include "data_builder.h"

namespace fs = std::filesystem;

// Function to create dataset with train/validation split
void create_dataset(const std::string& dataset_dir, const std::string& output_dir, 
                    float train_ratio, int frame_interval) {
    // Create main dataset directory if it doesn't exist
    if (!directory_exists(output_dir)) {
        create_directories(output_dir);
    }
    
    // Create train and val directories
    std::string train_dir = output_dir + "/train";
    std::string val_dir = output_dir + "/valid";
    
    if (!directory_exists(train_dir)) {
        create_directories(train_dir);
    }
    
    if (!directory_exists(val_dir)) {
        create_directories(val_dir);
    }
    
    // Create image and label subdirectories for both train and val
    std::string train_img_dir = train_dir + "/images";
    std::string train_label_dir = train_dir + "/labels";
    std::string val_img_dir = val_dir + "/images";
    std::string val_label_dir = val_dir + "/labels";
    
    if (!directory_exists(train_img_dir)) {
        create_directories(train_img_dir);
    }
    
    if (!directory_exists(train_label_dir)) {
        create_directories(train_label_dir);
    }
    
    if (!directory_exists(val_img_dir)) {
        create_directories(val_img_dir);
    }
    
    if (!directory_exists(val_label_dir)) {
        create_directories(val_label_dir);
    }
    
    // Create array of directories for process_video function
    std::vector<std::string> output_dirs = {
        train_img_dir,
        train_label_dir,
        val_img_dir,
        val_label_dir
    };
    
    // Get paths to annotations and videos directories
    std::string annotations_dir = dataset_dir + "/annotations";
    std::string videos_dir = dataset_dir + "/videos";
    
    if (!directory_exists(annotations_dir) || !directory_exists(videos_dir)) {
        std::cerr << "Error: Dataset directory structure is invalid. Expected annotations and videos directories." << std::endl;
        return;
    }
    
    // Process all video and annotation pairs in the dataset
    int processed_count = 0;
    
    // Iterate through all video files in the videos directory
    for (const auto& video_entry : fs::directory_iterator(videos_dir)) {
        if (!video_entry.is_regular_file() || video_entry.path().extension() != ".mov") {
            continue;
        }
        
        // Extract base name without extension (e.g., "location_video0")
        std::string base_name = video_entry.path().stem().string();
        std::string video_file = video_entry.path().string();
        
        // Find corresponding annotation file
        std::string annotation_file = annotations_dir + "/" + base_name + ".txt";
        
        if (!fs::exists(annotation_file)) {
            std::cerr << "Warning: No matching annotation file found for " << video_file << std::endl;
            continue;
        }
        
        std::cout << "Processing pair: " << std::endl;
        std::cout << "  Video: " << video_file << std::endl;
        std::cout << "  Annotation: " << annotation_file << std::endl;
        
        // Call process_video function to process this pair
        process_video(video_file, annotation_file, output_dirs, train_ratio, frame_interval);
        
        processed_count++;
    }
    
    std::cout << "Dataset creation complete. Processed " << processed_count << " video/annotation pairs." << std::endl;
    std::cout << "Train/validation split: " << (train_ratio * 100) << "% / " 
              << ((1 - train_ratio) * 100) << "%" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        std::cout << "Usage: " << argv[0] << " <input_dataset_dir> <output_dataset_dir> [train_ratio] [frame_interval]" << std::endl;
        std::cout << "Default train_ratio if not specified: 0.8 (80% train, 20% validation)" << std::endl;
        std::cout << "Default frame_interval if not specified: 10" << std::endl;
        return 1;
    }
    
    std::string input_dataset_dir = argv[1];  // Directory containing annotations and videos folders
    std::string output_dir = argv[2];         // Output directory for the processed dataset
    
    float train_ratio = 0.8f;  // Default value
    int frame_interval = 10;   // Default value
    
    // If train_ratio is provided as argument
    if (argc >= 4) {
        try {
            train_ratio = std::stof(argv[3]);
            if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
                std::cout << "Error: train_ratio must be between 0.0 and 1.0" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid train_ratio value. Must be a number between 0.0 and 1.0" << std::endl;
            return 1;
        }
    }
    
    // If frame_interval is provided as argument
    if (argc >= 5) {
        try {
            frame_interval = std::stoi(argv[4]);
            if (frame_interval <= 0) {
                std::cout << "Error: frame_interval must be a positive integer" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid frame_interval value. Must be a positive integer" << std::endl;
            return 1;
        }
    }
    
    create_dataset(input_dataset_dir, output_dir, train_ratio, frame_interval);
    
    return 0;
}