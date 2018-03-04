
#include "inferencemap.hpp"

#include "opencv2/opencv.hpp"
using namespace cv;

#define py_assert(condition)                          \
    do                                                \
    { /*printf("assert: %s\n", #condition) ;*/        \
        if (!(condition))                             \
        {                                             \
            failmsg("assert failed: %s", #condition); \
            return false;                             \
        }                                             \
    } while (0)

InferenceMap::InferenceMap()
    : arrobj(NULL), data(NULL)
{
}

bool InferenceMap::init(PyObject *pyobj)
{
    py_assert(pyobj != NULL);
    py_assert(PyArray_Check(pyobj));
    PyArrayObject *obj = (PyArrayObject *)pyobj;

    py_assert(PyArray_TYPE(obj) == NPY_FLOAT);
    py_assert(PyArray_NDIM(obj) == 3);

    const npy_intp *_sizes = PyArray_DIMS(obj);
    const npy_intp *_strides = PyArray_STRIDES(obj);
    intptr_t channels = _sizes[2];

    py_assert(channels == 5);

    Py_XDECREF(this->arrobj);

    intptr_t elemsize = 4;
    intptr_t height = _sizes[0];
    intptr_t width = _sizes[1];
    bool needcopy = (_strides[2] != elemsize) || (_strides[1] != elemsize * channels) || (_strides[0] != elemsize * channels * width);

    if (needcopy)
        obj = PyArray_GETCONTIGUOUS(obj);
    else
        Py_INCREF(obj);

    if (obj == NULL)
        return false;

    this->arrobj = obj;
    this->data = (InferencePixel *)PyArray_DATA(obj);
    this->height = height;
    this->width = width;

    // printf("strides: %d, %d\n", (int)((intptr_t)&at(1, 0) - (intptr_t)&at(0, 0)), (int)((intptr_t)&at(0, 1) - (intptr_t)&at(0, 0)));

    return true;
}

InferenceMap::~InferenceMap()
{
    Py_XDECREF(arrobj);
}

InferenceMap::InferenceMap(InferenceMap &obj)
{
    this->width = obj.width;
    this->height = obj.height;
    this->arrobj = obj.arrobj;
    this->data = obj.data;

    Py_XINCREF(this->arrobj);
}

InferenceMap &InferenceMap::operator=(InferenceMap &obj)
{
    Py_XDECREF(this->arrobj);
    this->width = obj.width;
    this->height = obj.height;
    this->arrobj = obj.arrobj;
    this->data = obj.data;
    Py_XINCREF(this->arrobj);

    return *this;
}
