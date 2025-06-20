#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Intrinsic matrix
    cv::Mat K = (cv::Mat_<float>(3, 3) << 2179.63, 0, 473.28,
                                          0, 1958.57, 527.62,
                                          0, 0, 1);
    
    // Extrinsic parameters (top-down orientation)
    cv::Mat R = (cv::Mat_<float>(3, 3) << 1, 0, 0,
                                          0, -1, 0,
                                          0, 0, -1);
    // Altitude in meters
    float h = 30.0;
    // Baseline displacement in meters
    float baseline = 5.0;

    // Projection matrices P1 and P2
    cv::Mat t1 = (cv::Mat_<float>(3, 1) << 0, 0, h);
    cv::Mat t2 = (cv::Mat_<float>(3, 1) << baseline, 0, h);

    cv::Mat P1, P2;
    cv::hconcat(R, t1, P1);
    cv::hconcat(R, t2, P2);

    P1 = K * P1;
    P2 = K * P2;

    // Corresponding points in both images
    cv::Mat points1 = (cv::Mat_<float>(2, 4) << 200, 400, 150, 700,
                                                 300, 500, 600, 200);
    cv::Mat points2 = (cv::Mat_<float>(2, 4) << 180, 380, 130, 680,
                                                 310, 510, 610, 210);
    
    // Triangulation
    cv::Mat points_4d;
    cv::triangulatePoints(P1, P2, points1, points2, points_4d);

    // Convert to 3D coordinates
    cv::Mat points_3d;
    cv::convertPointsFromHomogeneous(points_4d.t(), points_3d);

    // Convert to ground coordinates (Z is depth below drone)
    for (int i = 0; i < points_3d.rows; i++) {
        points_3d.at<cv::Vec3f>(i)[2] = h - points_3d.at<cv::Vec3f>(i)[2];
    }

    std::cout << "Reconstructed 3D Ground Coordinates:\n" << points_3d << std::endl;
    return 0;
}
