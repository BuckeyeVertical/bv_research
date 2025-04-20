#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

void createDirectoryIfNotExists(const fs::path& dirPath) {
    if (!fs::exists(dirPath)) {
        fs::create_directories(dirPath);
        std::cout << "Created directory: " << dirPath << std::endl;
    }
}

std::string getNewFileName(const fs::path& filePath) {
    // Extract location and video number from path
    std::vector<std::string> pathComponents;
    
    // Start from dataset and find components
    fs::path current = filePath;
    while (current.has_parent_path()) {
        std::string filename = current.filename().string();
        if (filename == "dataset" || filename == "annotations" || filename == "videos") {
            break;
        }
        
        pathComponents.push_back(filename);
        current = current.parent_path();
    }
    
    // Reverse to get correct order (location->video#->file)
    std::reverse(pathComponents.begin(), pathComponents.end());
    
    // We need location and video# for the new name
    if (pathComponents.size() >= 2) {
        std::string location = pathComponents[0];
        std::string videoNum = pathComponents[1];
        
        // Remove any non-alphanumeric characters from location and videoNum
        location.erase(std::remove_if(location.begin(), location.end(), 
                      [](char c) { return !std::isalnum(c); }), location.end());
        videoNum.erase(std::remove_if(videoNum.begin(), videoNum.end(), 
                      [](char c) { return !std::isalnum(c); }), videoNum.end());
        
        return location + "_" + videoNum;
    }
    
    return filePath.stem().string(); // Fallback to original name without extension
}

int main(int argc, char* argv[]) {
    // Check if a base directory path was provided as a command-line argument
    std::string baseDir = (argc > 1) ? argv[1] : "dataset";
    
    // Output directories
    fs::path outputBaseDir = "data";
    fs::path outputAnnotationsDir = outputBaseDir / "annotations";
    fs::path outputVideosDir = outputBaseDir / "videos";
    
    // Create output directories if they don't exist
    createDirectoryIfNotExists(outputBaseDir);
    createDirectoryIfNotExists(outputAnnotationsDir);
    createDirectoryIfNotExists(outputVideosDir);
    
    // Counters for summary
    int videoCount = 0;
    int annotationCount = 0;
    
    try {
        // Process annotations
        fs::path annotationsPath = fs::path(baseDir) / "annotations";
        if (fs::exists(annotationsPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(annotationsPath)) {
                if (fs::is_regular_file(entry) && entry.path().filename() == "annotations.txt") {
                    std::string newName = getNewFileName(entry.path()) + ".txt";
                    fs::path destPath = outputAnnotationsDir / newName;
                    
                    std::cout << "Moving: " << entry.path() << " to " << destPath << std::endl;
                    
                    // Copy the file to the new location with the new name
                    fs::copy_file(entry.path(), destPath, fs::copy_options::overwrite_existing);
                    annotationCount++;
                }
            }
        } else {
            std::cerr << "Annotations directory not found: " << annotationsPath << std::endl;
        }
        
        // Process videos
        fs::path videosPath = fs::path(baseDir) / "videos";
        if (fs::exists(videosPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(videosPath)) {
                if (fs::is_regular_file(entry) && entry.path().filename() == "video.mov") {
                    std::string newName = getNewFileName(entry.path()) + ".mov";
                    fs::path destPath = outputVideosDir / newName;
                    
                    std::cout << "Moving: " << entry.path() << " to " << destPath << std::endl;
                    
                    // Copy the file to the new location with the new name
                    fs::copy_file(entry.path(), destPath, fs::copy_options::overwrite_existing);
                    videoCount++;
                }
            }
        } else {
            std::cerr << "Videos directory not found: " << videosPath << std::endl;
        }
        
        std::cout << "\nProcessing complete!" << std::endl;
        std::cout << "Total annotation files processed: " << annotationCount << std::endl;
        std::cout << "Total video files processed: " << videoCount << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}