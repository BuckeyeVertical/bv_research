// 1) take in annotation txt file and video mov file.
// 2) run annotation parser and initialize dataset, save 80% of annotation files to dataset/train/labels and 20% to dataset/valid/labels
// 3) run video parser and save 80% of image files to dataset/train/images and 20% to dataset/valid/images

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "structs.h"
#include "helpers.h"
#include "sdd_parser.h"
#include "data_builder.h"

// Function to create dataset with train/validation split
void create_dataset(const std::string& annotation_file, const std::string& video_file, 
                    const std::string& dataset_dir, float train_ratio) {
    // Create main dataset directory if it doesn't exist
    if (!directory_exists(dataset_dir)) {
        create_directories(dataset_dir);
    }
    
    // Create train and val directories
    std::string train_dir = dataset_dir + "/train";
    std::string val_dir = dataset_dir + "/valid";
    
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
    
    // Process the video and create the dataset
    std::cout << "Creating dataset in " << dataset_dir << std::endl;
    std::cout << "Train/validation split: " << (train_ratio * 100) << "% / " 
              << ((1 - train_ratio) * 100) << "%" << std::endl;
    
    // Call process_video function to create the dataset
    process_video(video_file, annotation_file, output_dirs, train_ratio);
}

int main(int argc, char* argv[]) {
    if (argc != 4 && argc != 5) {
        std::cout << "Usage: " << argv[0] << " <annotation_file> <video_file> <output_dataset_dir> [train_ratio]" << std::endl;
        std::cout << "Default train_ratio if not specified: 0.8 (80% train, 20% validation)" << std::endl;
        return 1;
    }
    
    std::string annotation_file = argv[1];
    std::string video_file = argv[2];
    std::string output_dir = argv[3];
    
    float train_ratio = 0.8f; // Default value
    
    // If train_ratio is provided as argument
    if (argc == 5) {
        try {
            train_ratio = std::stof(argv[4]);
            if (train_ratio <= 0.0f || train_ratio >= 1.0f) {
                std::cout << "Error: train_ratio must be between 0.0 and 1.0" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid train_ratio value. Must be a number between 0.0 and 1.0" << std::endl;
            return 1;
        }
    }
    
    create_dataset(annotation_file, video_file, output_dir, train_ratio);
    
    return 0;
}