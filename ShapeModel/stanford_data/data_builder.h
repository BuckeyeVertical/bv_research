#ifndef DATA_BUILDER_H
#define DATA_BUILDER_H

#include <string>

void create_dataset(const std::string& annotation_file, const std::string& video_file, 
                    const std::string& dataset_dir, float train_ratio);

#endif