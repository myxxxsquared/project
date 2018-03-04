
#ifndef PYTHONHEADER_HEADER
#define PYTHONHEADER_HEADER

#ifndef MYLIBRARY_USE_IMPORT
#define NO_IMPORT
#endif

#define PY_ARRAY_UNIQUE_SYMBOL MYLIBRARY_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL MYLIBRARY_UFUNC_API

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <opencv2/opencv.hpp>

using namespace cv;

inline int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
  public:
    inline PyAllowThreads() : _state(PyEval_SaveThread()) {}
    inline ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }

  private:
    PyThreadState *_state;
};

class PyEnsureGIL
{
  public:
    inline PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    inline ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }

  private:
    PyGILState_STATE _state;
};

#endif /* PYTHONHEADER_HEADER */
