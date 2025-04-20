#ifndef DATA_BUILDER_H
#define DATA_BUILDER_H

#include <string>

void create_dataset(const std::string& dataset_dir, const std::string& output_dir, 
                    float train_ratio, int frame_interval);

#endif