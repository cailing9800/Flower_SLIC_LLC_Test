#ifndef DICTIONARY_H_INCLUDED
#define DICTIONARY_H_INCLUDED

#include <opencv2/opencv.hpp>

void llc_gen_dictionary(const std::string& dictionaryFile, cv::Mat &dictionary);

#endif // DICTIONARY_H_INCLUDED
