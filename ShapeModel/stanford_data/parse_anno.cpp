#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct Annotation {
    std::string data;  // Will store "label_number x1 y1 x2 y2"
};

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

int main() {
    // Open the file
    std::ifstream file("bookstore_vid0_annotations.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
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
    // Get base filename without extension
    std::string baseFilename = getBaseFilename("bookstore_vid0_annotations.txt");

    // Write each frame's annotations to a separate file
    for (const auto& pair : hashtable) {
        int frameNumber = pair.first;
        const std::vector<Annotation>& annotations = pair.second;
        
        // Create output filename: baseFilename_frameNumber.txt
        std::ostringstream outputFilenameStream;
        outputFilenameStream << baseFilename << "_" << frameNumber << ".txt";
        std::string outputFilename = outputFilenameStream.str();
        
        // Open output file
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            std::cerr << "Error opening output file: " << outputFilename << std::endl;
            continue;  // Skip to next frame
        }
        
        // Write annotations to file (one per line)
        for (const auto& annotation : annotations) {
            outputFile << annotation.data << std::endl;
        }
        
        // Close output file
        outputFile.close();
        
        std::cout << "Created file: " << outputFilename << " with " 
                  << annotations.size() << " annotation(s)" << std::endl;
    }

    std::cout << "Processed " << hashtable.size() << " frames in total." << std::endl;
    
    return 0;
}