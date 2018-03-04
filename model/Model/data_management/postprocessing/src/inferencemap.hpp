#ifndef INFERENCEMAP_HEADER
#define INFERENCEMAP_HEADER

#include "pythonheader.hpp"

struct InferencePixel
{
    float tcl, radius, cos, sin, tr;
};

class InferenceMap
{
  public:
    int width, height;
    PyArrayObject *arrobj;
    const InferencePixel *data;

    InferenceMap();
    InferenceMap(InferenceMap &obj);
    ~InferenceMap();
    InferenceMap &operator=(InferenceMap &obj);
    bool init(PyObject *obj);

    const inline InferencePixel &at(int x, int y) const
    {
        return data[y * width + x];
    }

    const inline InferencePixel &at(Point2i pt) const
    {
        return this->at(pt.x, pt.y);
    }

    // inline void savetest() const
    // {
    //     Mat mat;
    //     mat.create(height, width, CV_8UC1);
    //     for (int i = 0; i < height; ++i)
    //         for (int j = 0; j < width; ++j)
    //             mat.at<uchar>(i, j) = at(j, i).tcl > 0.5 ? 255 : 0;
    //     imwrite("test.png", mat);
    // }
};

#endif /* INFERENCEMAP_HEADER */