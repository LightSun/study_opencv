/*
* Copyright 2021 Zuru Tech HK Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* istributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <filesystem>
#include <sstream>
#include <fstream>

/*
#include <CLI/CLI.hpp>
#include "libs/magic_enum/include/magic_enum/magic_enum.hpp"
*/

#include <opencv2/opencv.hpp>

#include <PillowResize/PillowResize.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ImPlatformSim.h"
#include "ImagingSim.h"

namespace fs = std::filesystem;
/*
int main(int argc, char** argv)
{
    CLI::App app{"Resize an image using Pillow resample"};

    fs::path image_path, out_path;
    app.add_option("-i,--input", image_path, "Input image path.")->mandatory();
    app.add_option("-o,--output", out_path,
                   "Output image folder. \n"
                   "If empty the output image will be saved in the "
                   "input directory with the suffix '_resized'.");

    uint width = 0, height = 0;
    app.add_option("--ow,--width", width,
                   "Output image width. (default: 0).\n"
                   "If you want to set the output image size you need to pass "
                   "both width and height.");
    app.add_option("--oh,--height", height,
                   "Output image height. (default: 0)");

    double fx = 0., fy = 0.;
    app.add_option("--horizontal_scale_factor,--fx", fx,
                   "Scale factor along the horizontal axis. (default: 0).\n"
                   "Output image size will automatically computed as "
                   "image_width*horizontal_scale_factor.\n"
                   "If you want to set the scale you need to pass "
                   "both horizontal_scale_factor and vertical_scale_factor.");
    app.add_option("--vertical_scale_factor,--fy", fy,
                   "Scale factor along the vertical axis. (default: 0)\n"
                   "Output image size will automatically computed as "
                   "image_height*vertical_scale_factor.");

    // enum_names return list of name sorted by enum values.
    constexpr auto& interpolation_names =
        magic_enum::enum_names<PillowResize::InterpolationMethods>();
    std::stringstream interpolation_help;
    interpolation_help << "Interpolation method. \nValid values in:\n";
    for (size_t i = 0; i < interpolation_names.size(); ++i) {
        interpolation_help << interpolation_names[i] << " -> " << i
                           << std::endl;
    }

    auto transformArgToEnum = [](const std::string& s) -> std::string {
        auto isInt = [](const std::string& s, int& value) -> bool {
            try {
                value = std::stoi(s);
                return true;
            }
            catch (const std::exception&) {
                return false;
            }
        };

        std::optional<PillowResize::InterpolationMethods> interpolation;
        int arg_converted;
        if (isInt(s, arg_converted)) {
            interpolation =
                magic_enum::enum_cast<PillowResize::InterpolationMethods>(
                    arg_converted);
        }
        else {
            interpolation =
                magic_enum::enum_cast<PillowResize::InterpolationMethods>(s);
        }
        if (!interpolation.has_value()) {
            throw CLI::ValidationError("Interpolation method not valid.");
        }
        return std::to_string(magic_enum::enum_integer(interpolation.value()));
    };

    PillowResize::InterpolationMethods interpolation =
        PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR;
    app.add_option("-m, --method", interpolation, interpolation_help.str())
        ->required()
        ->transform(transformArgToEnum);

    CLI11_PARSE(app, argc, argv);

    // Check if out_path is empty.
    // If empty, create out_path from image_path with the suffix _resized.
    if (out_path.empty()) {
        out_path = image_path.parent_path() /
                   (image_path.stem().string() + "_" +
                    std::string(magic_enum::enum_name(interpolation)) +
                    image_path.extension().string());
    }

    auto out_parent_path = out_path.parent_path();
    // If image_path is a relative path to the current directory,
    // its parent path will be empty.
    // This is necessary in case the library is installed and the binary
    // is run using its full path (installation folder) from a folder with images.
    if (out_parent_path.empty()) {
        out_path = fs::current_path() / out_path;
        out_parent_path = fs::current_path();
    }
    if (!fs::exists(out_parent_path)) {
        fs::create_directories(out_parent_path);
    }

    cv::Mat input = cv::imread(image_path.string(), cv::IMREAD_ANYCOLOR);

    cv::Size out_size;
    if (width != 0 && height != 0) {
        out_size = cv::Size(width, height);
    }
    else if (fx > 0 && fy > 0) {
        out_size = cv::Size(static_cast<int>(input.size().width * fx),
                            static_cast<int>(input.size().height * fy));
    }
    else {
        std::cout << "You need to set the output size. \n"
                     "Set both width and height or both fx and fy."
                  << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat out = PillowResize::resize(input, out_size, interpolation);

    cv::imwrite(out_path.string(), out);

    return EXIT_SUCCESS;
}
*/

//---------------
template<typename T>
static bool saveAsPilImage(const std::string& fileName, const T* ptr, int count){
    std::string str;
    str.reserve(count);
    for(int i = 0 ; i < count; ++i){
        str += std::to_string(ptr[i]);
#ifdef _WIN32
        str += "\r\n";
#else
        str += "\n";
#endif
    }
    std::ofstream outfile(fileName);
    if (!outfile.is_open())
    {
       return false;
    }
    outfile.write(str.data(), str.length());
    outfile.flush();
    outfile.close();
    return true;
}

static void test1();
int main(int argc, char** argv)
{
    test1();
    return 0;
}

static cv::Mat processImage(const cv::Mat& img_split) {
    cv::Mat img;
    cv::cvtColor(img_split, img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
//    if(pillowResize(img, resized_img, "RGB", 224, 224,
//                    IMAGING_TRANSFORM_BILINEAR) !=0){
//        fprintf(stderr, "pillowResize error.\n");
//        abort();
//    }

    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
    //
    std::vector<float> mean = {0.29003, 0.29385, 0.31377};
    std::vector<float> std = {0.18866, 0.19251, 0.19958};
    for (int c = 0; c < float_img.channels(); ++c) {
        for (int i = 0; i < float_img.rows; ++i) {
            for (int j = 0; j < float_img.cols; ++j) {
                float& pixel = float_img.at<cv::Vec3f>(i, j)[c];
                pixel = (pixel - static_cast<float>(mean[c])) / static_cast<float>(std[c]);
            }
        }
    }

    return float_img;
}

static void processImage2(const cv::Mat& img_split) {
    printf("img_split.depth() = %d\n", img_split.depth());
    printf("img_split.channels() = %d\n", img_split.channels());
    cv::Mat img;
    cv::cvtColor(img_split, img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
//    if(pillowResize(img, resized_img, "RGB", 224, 224,
//                    IMAGING_TRANSFORM_BILINEAR) !=0){
//        fprintf(stderr, "pillowResize error.\n");
//        abort();
//    }
//    PillowResize::InterpolationMethods interpolation =
//        PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR;
//    resized_img = PillowResize::resize(img, cv::Size(224, 224), interpolation);
    resized_img.convertTo(resized_img, CV_32FC3, 1 / 255.0);
    //
    std::vector<float> mean = {0.29003, 0.29385, 0.31377};
    std::vector<float> std = {0.18866, 0.19251, 0.19958};
    std::vector<cv::Mat> rgbs(3);
    cv::split(resized_img, rgbs);

    for (auto i = 0; i < (int)rgbs.size(); i++)
    {
        rgbs[i].convertTo(rgbs[i], CV_32FC1, 1,
                          (0.0 - mean[i]) / std[i]);
    }
    int imgSize = 224 * 224;
    std::vector<float> buffer;
    buffer.resize(imgSize * 3);
    for(int i = 0 ; i < rgbs.size() ; ++i){
        float* dPtr = buffer.data() + i * imgSize;
        memcpy(dPtr, rgbs[i].ptr<float>(0), imgSize * sizeof(float));
    }
    std::string outDir = "/home/heaven7/heaven7/work/SE/MedQA/MN/us-qc/h7";
    saveAsPilImage<float>(outDir + "/normal_cpp_pil.txt",
                          buffer.data(), buffer.size());
}

static void test1(){
    std::string imgDir = "/media/heaven7/Elements_SE/study/work/MedQA/meinian/imgs/器官图像";
    std::string image_path = imgDir + "/qianliexian.png";
    std::string out_path = imgDir + "/qianliexian_pil_resize.png";
    //double fx = 0., fy = 0.;
    int width = 224;
    int height = 224;
    //
    cv::Mat img_raw = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Size size(width, height);
    if(size.width > 0 && size.height > 0 && (image.cols != size.width
                                             || image.rows != size.height))
    {
        //cv::resize(image, image, size, 0, 0, cv::INTER_NEAREST);
//        PillowResize::InterpolationMethods interpolation =
//            PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR;
//        image = PillowResize::resize(image, size, interpolation);
        if(pillowResize(image, image, "RGB", width, height, IMAGING_TRANSFORM_BILINEAR) !=0){
            fprintf(stderr, "pillowResize error.\n");
            return;
        }
    }

    image.convertTo(image, CV_32FC3, 1 / 255.0);

    int img_h = size.height;
    int img_w = size.width;
    float *img_data = (float *) image.data;
    int hw = img_h * img_w;
    //double scalefactor = 1.0 / 255;
    //nchw
    std::vector<float> buffer;
    buffer.resize(img_h * img_w * 3);
    std::vector<float> mean = {0.29003, 0.29385, 0.31377};
    std::vector<float> std = {0.18866, 0.19251, 0.19958};
    float* hostDataBuffer = static_cast<float*>(buffer.data());
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                hostDataBuffer[c * hw + h * img_w + w] = (*img_data - mean[c]) / std[c];
                img_data++;
            }
        }
    }
    //
    processImage2(img_raw);

    std::string outDir = "/home/heaven7/heaven7/work/SE/MedQA/MN/us-qc/h7";
    saveAsPilImage<float>(outDir + "/normal_cpp_pil.txt",
                          buffer.data(), buffer.size());
//    saveAsPilImage<float>(outDir + "/normal_cpp_pil.txt",
//                          img.ptr<float>(0), buffer.size());
}
