
#ifndef POSTPROCESSING_HEADER
#define POSTPROCESSING_HEADER

#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

#include "inferencemap.hpp"

struct ProcessConfig
{
    float t_tcl, t_tr, t_delta, t_rad;
    float fewest_tcl_ratio, smallest_area_ratio, smallest_area_ratio_tcl;
    float radius_scaling;
    int fewest_tcl;
    bool load_from(PyObject *obj);
};

struct RegionInfo
{
    Mat region;
    float avg_r, avg_x, avg_y, avg_cos, avg_sin;
    int ptnumber, area;
    vector<vector<Point>> contours;
};

class PostProcessor
{
  public:
    ProcessConfig config;
    InferenceMap map;
    Mat search_mark;
    Mat result;
    Mat trmap;

    vector<RegionInfo> regions;

    bool postprocess();

    static vector<Point2i> generate_random_vector(int rows, int cols);
    void search_contour(Point2i pt);
};

#endif
