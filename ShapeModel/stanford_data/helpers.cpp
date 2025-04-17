#include <string>
#include <sys/stat.h>
#include <iostream>

#include "helpers.h"

//annotation helpers **
// Helper function to convert label text to numeric code
int getLabelCode(const std::string& label) {
    // Convert label to lowercase for case-insensitive comparison
    std::string lowerLabel = label;
    std::transform(lowerLabel.begin(), lowerLabel.end(), lowerLabel.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
    
    if (lowerLabel == "pedestrian" || lowerLabel == "skateboarder" || lowerLabel == "skater") {
        return 1;
    } else if (lowerLabel == "car" || lowerLabel == "cart") {
        return 0;
    } else if (lowerLabel == "cyclist" || lowerLabel == "biker") {
        return 2;
    } else if (lowerLabel == "bus") {
        return 3;
    } else {
        // Print the unknown label for debugging
        std::cout << "Unknown label: \"" << label << "\"" << std::endl;
        return -1;
    }
}

// Helper function to get filename without extension
std::string getBaseFilename(const std::string& filepath) {
    // Find the last slash or backslash
    size_t lastSlash = filepath.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);
    
    // Find the last dot (extension)
    size_t lastDot = filename.find_last_of(".");
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}

//video helpers **
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