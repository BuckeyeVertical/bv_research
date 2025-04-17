#ifndef SDD_PARSER_H
#define SDD_PARSER_H

#include <string>

void process_video(const std::string& video_path, const std::string& text_path,
                  const std::vector<std::string>& output_dirs, float split);

#endif