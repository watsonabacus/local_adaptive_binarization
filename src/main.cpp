/*
 * C++ sample to demonstrate Niblack thresholding.
 */

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "niblack_thresholding.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

Mat_<uchar> src;
int k_ = 8;
int blockSize_ = 11;
int type_ = THRESH_BINARY;
int method_ = BINARIZATION_NIBLACK;

void on_trackbar(int, void*);
void onMouse( int event, int x, int y, int flags, void* userdata );

int main(int argc, char** argv)
{
    // read gray-scale image
    if(argc != 2)
    {
        cout << "Usage: ./niblack_thresholding [IMAGE]\n";
        return 1;
    }
    const char* filename = argv[1];
    src = imread(filename, IMREAD_GRAYSCALE);
    int width = src.cols/2;
    int height = src.rows/2;
    namedWindow("Source", WINDOW_NORMAL);
    resizeWindow("Source", width,height);
    setMouseCallback( "Source", onMouse, &src );
    imshow("Source", src);

    namedWindow("Niblack", WINDOW_NORMAL);
    resizeWindow("Niblack", width,height);
    setMouseCallback( "Niblack", onMouse, &src );
    createTrackbar("k", "Niblack", &k_, 20, on_trackbar);
    createTrackbar("blockSize", "Niblack", &blockSize_, 30, on_trackbar);
    createTrackbar("method", "Niblack", &method_, 3, on_trackbar);
    createTrackbar("threshType", "Niblack", &type_, 4, on_trackbar);
    on_trackbar(0, 0);
    waitKey(0);

    return 0;
}
void onMouse( int event, int x, int y, int flags, void* userdata )
{
//    if( event != CV_EVENT_LBUTTONDOWN )
//            return;
    Mat_<uchar> *img = (Mat_<uchar> *)userdata;
    Point pt = Point(x,y);
    std::cout<<"x="<<pt.x<<"\t y="<<pt.y<<"\t value="<<int(img->at<uchar>(y, x))<<"\n";

}

void on_trackbar(int, void*)
{
    double k = static_cast<double>(k_-10)/10;                 // [-1.0, 1.0]
    int blockSize = 2*(blockSize_ >= 1 ? blockSize_ : 1) + 1; // 3,5,7,...,61
    int type = type_;  // THRESH_BINARY, THRESH_BINARY_INV,
                       // THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
    int method = method_; //BINARIZATION_NIBLACK, BINARIZATION_SAUVOLA, BINARIZATION_WOLF, BINARIZATION_NICK
    Mat dst;
    niBlackThreshold(src, dst, 255, type, blockSize, k, method);
    imshow("Niblack", dst);
}
