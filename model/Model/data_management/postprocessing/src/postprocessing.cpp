
#include "inferencemap.hpp"

#include "postprocessing.hpp"

#include <cmath>

vector<Point2i> PostProcessor::generate_random_vector(int rows, int cols)
{
    vector<Point2i> rowlist;
    rowlist.resize(rows * cols);
    int k = 0;
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < rows; ++j)
            rowlist.at(k++) = Point2i(i, j);
    random_shuffle(rowlist.begin(), rowlist.end());
    return rowlist;
}

void PostProcessor::search_contour(Point2i pt)
{
    InferencePixel sp = map.at(pt);

    // vector<Point2i> points;
    deque<Point2i> to_search;
    // points.push_back(pt);
    float sum_radius = sp.radius;
    // float sum_x = pt.x;
    // float sum_y = pt.y;
    // float sum_cos = sp.cos;
    // float sum_sin = sp.sin;
    int ptnumber = 1;
    to_search.push_back(pt);
    search_mark.at<uchar>(pt) = 255;

    int search_distance = int(pow(sp.radius / 5.0, 1.0)) + 2;
    // printf("search_distance: %d\n", search_distance);
    // search_distance = 1;

    Mat region;
    region.create(map.height, map.width, CV_8UC1);
    region.setTo(Scalar((unsigned char)0));

    Mat tclregion;
    tclregion.create(map.height, map.width, CV_8UC1);
    tclregion.setTo(Scalar((unsigned char)0));

    while (to_search.size())
    {
        Point2i cur = to_search.front();
        to_search.pop_front();

        InferencePixel cp = map.at(cur);
        circle(region, cur, (int)(cp.radius * config.radius_scaling), Scalar((unsigned char)255), -1);
        tclregion.at<uchar>(cur) = 255;

        int xmin = std::max(0, cur.x - search_distance);
        int xmax = std::min(map.width - 1, cur.x + search_distance);
        int ymin = std::max(0, cur.y - search_distance);
        int ymax = std::min(map.height - 1, cur.y + search_distance);

        for (int y = ymin; y <= ymax; y++)
        {
            for (int x = xmin; x <= xmax; x++)
            {
                Point2i curpt{x, y};
                if (search_mark.at<uchar>(curpt))
                    continue;
                InferencePixel np = map.at(curpt);
                if (np.tr > config.t_tr && np.tcl > config.t_tcl)
               {
                    // if (!(abs(np.radius - cp.radius) < config.t_rad * cp.radius))
                    // {
                    //     // printf("abs(np.radius - cp.radius) < config.t_rad * cp.radius\n");
                    //     continue;
                    // }
                    // if (!(abs(np.cos - cp.cos) < config.t_delta))
                    // {
                    //     // printf("abs(np.cos - cp.cos) < config.t_delta\n");
                    //     continue;
                    // }
                    // if (!(abs(np.sin - cp.sin) < config.t_delta || 2 - abs(np.sin) - abs(cp.sin) < config.t_delta))
                    // {
                    //     // printf("abs(np.sin - cp.sin) < config.t_delta\n");
                    //     continue;
                    // }
                    search_mark.at<uchar>(curpt) = 255;
                    // points.push_back(curpt);
                    to_search.push_back(curpt);
                    sum_radius += np.radius;
                    // sum_x += x;
                    // sum_y += y;
                    // sum_cos += np.cos;
                    // sum_sin += np.sin;
                    ptnumber++;
                }
            }
        }
    }

    // printf("%d\n", i);

    sum_radius /= ptnumber;
    // sum_x /= ptnumber;
    // sum_y /= ptnumber;
    // sum_sin /= ptnumber;
    // sum_cos /= ptnumber;

    int area_region = countNonZero(region);
    Mat andtr, andtcl;
    bitwise_and(region, trmap, andtr);
    bitwise_and(tclregion, trmap, andtcl);
    int area_text_region = countNonZero(andtr);
    int area_text_region_tcl = countNonZero(andtcl);

    // if (ptnumber < config.fewest_tcl)
    // {
    //     // printf("ptnumber < config.fewest_tcl\n");
    //     return;
    // }
    if ((float)ptnumber / sum_radius / sum_radius < config.fewest_tcl_ratio)
    {
        // printf("(float)ptnumber / sum_radius / sum_radius < config.fewest_tcl_ratio\n");
        return;
    }
    if ((float)area_text_region / area_region < config.smallest_area_ratio)
    {
        // printf("(float)area_text_region / area_region < config.smallest_area_ratio\n");
        return;
    }

    // if ((float)area_text_region_tcl / ptnumber < config.smallest_area_ratio_tcl)
    // {
    //     // printf("(float)area_text_region / area_region < config.smallest_area_ratio\n");
    //     return;
    // }

    regions.emplace_back();
    auto &r = regions.back();
    // r.avg_x = sum_x;
    // r.avg_y = sum_y;
    // r.avg_r = sum_radius;
    // r.avg_cos = sum_cos;
    // r.avg_sin = sum_sin;
    // r.region = region;
    // r.ptnumber = ptnumber;
    // r.area = area_text_region;
    vector<Vec4i> hierarchy;
    findContours(region, r.contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1);
    // printf("contours in single: %d\n", (int)r.contours.size());
}

bool PostProcessor::postprocess()
{
    search_mark.create(map.height, map.width, CV_8UC1);
    search_mark.setTo(Scalar((unsigned char)0));

    trmap.create(map.height, map.width, CV_8UC1);

    // #pragma omp parallel for
    for (int i = 0; i < map.height; ++i)
    {
        uchar *tr = &trmap.at<uchar>(i, 0);
        const InferencePixel *pix = &map.at(0, i);
        for (int j = 0; j < map.width; ++j, ++pix, ++tr)
            *tr = pix->tr > config.t_tr ? 255 : 0;
    }

    vector<Point2i> ptlist = generate_random_vector(map.height, map.width);
    for (Point2i pt : ptlist)
    {
        if (map.at(pt).tcl > config.t_tcl && !search_mark.at<uchar>(pt))
        {
            search_contour(pt);
            // break;
        }
    }

    return true;
}

#define READ_FLOAT(name)                                           \
    do                                                             \
    {                                                              \
        PyObject *o = PyObject_GetAttrString(obj, #name);          \
        if (o == NULL)                                             \
            return false;                                          \
        this->name = PyFloat_AsDouble(o);                          \
        /*fprintf(stderr, "%s: %f\n", #name, (float)this->name);*/ \
    } while (0)

#define READ_INT(name)                                           \
    do                                                           \
    {                                                            \
        PyObject *o = PyObject_GetAttrString(obj, #name);        \
        if (o == NULL)                                           \
            return false;                                        \
        this->name = PyLong_AsLong(o);                           \
        /*fprintf(stderr, "%s: %d\n", #name, (int)this->name);*/ \
    } while (0)

bool ProcessConfig::load_from(PyObject *obj)
{
    READ_FLOAT(t_tcl);
    READ_FLOAT(t_tr);
    READ_FLOAT(t_delta);
    READ_FLOAT(t_rad);
    READ_FLOAT(fewest_tcl_ratio);
    READ_FLOAT(smallest_area_ratio);
    READ_FLOAT(smallest_area_ratio_tcl);
    READ_FLOAT(radius_scaling);
    READ_INT(fewest_tcl);

    return true;
}
