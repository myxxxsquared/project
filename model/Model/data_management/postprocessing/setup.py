
from distutils.core import setup, Extension

import os
import numpy

envpath = os.environ["CONDA_PREFIX"]
numpypath = numpy.__path__[0]

module_postprocessing = Extension(
    '_postprocessing',
    sources=[
        'src/inferencemap.cpp',
        'src/postprocessing.cpp',
        'src/py_postprocessing.cpp'],
    include_dirs=[
        os.path.join(envpath, "include"),
        os.path.join(numpypath, "core/include"), ],
    library_dirs=[
        os.path.join(envpath, "lib")],
    libraries=[
        "opencv_aruco",
        "opencv_bgsegm",
        "opencv_bioinspired",
        "opencv_calib3d",
        "opencv_ccalib",
        "opencv_core",
        "opencv_datasets",
        "opencv_dnn",
        "opencv_dpm",
        "opencv_face",
        "opencv_features2d",
        "opencv_flann",
        "opencv_fuzzy",
        "opencv_hdf",
        "opencv_highgui",
        "opencv_imgcodecs",
        "opencv_imgproc",
        "opencv_line_descriptor",
        "opencv_ml",
        "opencv_objdetect",
        "opencv_optflow",
        "opencv_photo",
        "opencv_plot",
        "opencv_reg",
        "opencv_rgbd",
        "opencv_saliency",
        "opencv_shape",
        "opencv_stereo",
        "opencv_stitching",
        "opencv_structured_light",
        "opencv_superres",
        "opencv_surface_matching",
        "opencv_text",
        "opencv_tracking",
        "opencv_videoio",
        "opencv_video",
        "opencv_videostab",
        "opencv_xfeatures2d",
        "opencv_ximgproc",
        "opencv_xobjdetect",
        "opencv_xphoto", ],
    depends=["opencv", "numpy"],
    extra_compile_args=['-O3', '-ggdb', '-std=gnu++11', '-Wall'],
    extra_link_args=['-O3', '-ggdb', '-Wall'],)

setup(name='postprocessing',
      version='1.0',
      description='postprocessing',
      ext_modules=[module_postprocessing])
