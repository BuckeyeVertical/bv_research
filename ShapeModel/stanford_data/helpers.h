#ifndef HELPERS_H
#define HELPERS_H

#include <string>

int getLabelCode(const std::string& label);
std::string getBaseFilename(const std::string& filepath);
void create_directories(const std::string& path);
bool directory_exists(const std::string& path);

#endif