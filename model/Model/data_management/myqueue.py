import multiprocessing
import queue
import threading
import numpy as np
import pickle
import os
import time
import gzip

import ctypes

import functools
import mmap


class SharedMemoryQueue():
    def __init__(self, maxsize, imgshape, mapshape):
        self.readq = multiprocessing.Queue(maxsize)
        self.writeq = multiprocessing.Queue(maxsize)

        self.imgshape = imgshape
        self.mapshape = mapshape
        self.weightshape = (self.imgshape[0],self.imgshape[1],self.imgshape[2],1)

        self.imgsize = functools.reduce(lambda x, y: x * y, imgshape, 1)
        self.imgsizebyte = self.imgsize
        self.mapsize = functools.reduce(lambda x, y: x * y, mapshape, 1)
        self.mapsizebyte = self.mapsize * 4
        self.weightsize = functools.reduce(lambda x, y: x * y, self.weightshape, 1)
        self.weightsizebyte = self.weightsize * 4

        print(self.imgsize, self.mapsize, self.weightsizebyte)

        # self.bufferimg = [multiprocessing.RawArray(
        #     'B', self.imgsize) for _ in range(maxsize)]
        # self.buffermap = [multiprocessing.RawArray(
        #     'f', self.mapsize) for _ in range(maxsize)]

        self.bufferimg = [mmap.mmap(-1, self.imgsizebyte) for _ in range(maxsize)]
        self.buffermap = [mmap.mmap(-1, self.mapsizebyte) for _ in range(maxsize)]
        self.bufferweight = [mmap.mmap(-1, self.weightsizebyte) for _ in range(maxsize)]

        self.imgtype = ctypes.c_int8 * self.imgsize
        self.maptype = ctypes.c_float * self.mapsize
        self.weighttype = ctypes.c_float * self.weightsize

        for i in range(maxsize):
            self.writeq.put(i)

    def put(self, data, path):
        _, img, maps, _, weights = data

        # print(maps.dtype)
        maps = maps.astype(np.float32)
        weights = weights.astype(np.float32)

        if path:
            for i in range(len(data[0])):
                pickle.dump(compress(data[0][i], data[1][i, :, :], data[2][i, :, :], data[3], data[4][i, :, :]),
                            gzip.open(os.path.join(path, str(hash(data[0][i])) + str(time.time()) + '.bin.gz'), 'wb'))

        i = self.writeq.get()
        # ctypes.memmove(
        #     self.bufferimg[i], np.ctypeslib.as_ctypes(img), self.imgsize)
        # ctypes.memmove(
        #     self.buffermap[i], np.ctypeslib.as_ctypes(maps), self.mapsize*4)
        self.bufferimg[i].seek(0)
        self.bufferimg[i].write(img.tobytes())
        self.buffermap[i].seek(0)
        self.buffermap[i].write(maps.tobytes())
        self.bufferweight[i].seek(0)
        self.bufferweight[i].write(weights.tobytes())
        self.readq.put(i)

    def get(self):
        i = self.readq.get()
        # imgbuf = self.imgtype()
        # mapbuf = self.maptype()

        # ctypes.memmove(imgbuf, self._bufferimg[i], self.imgsize)
        # ctypes.memmove(mapbuf, self._buffermap[i], self.mapsize*4)
        self.bufferimg[i].seek(0)
        self.buffermap[i].seek(0)
        self.bufferweight[i].seek(0)

        result = ("",
            np.frombuffer(self.bufferimg[i].read(), dtype=np.uint8).reshape(self.imgshape),
            np.frombuffer(self.buffermap[i].read(), dtype=np.float32).reshape(self.mapshape),
            None,
            np.frombuffer(self.bufferweight[i].read(), dtype=np.float32).reshape(self.weightshape))
        self.writeq.put(i)
        return result


def compress(name, img, maps, cnt, weight):
    ins = [name]
    ins.append(np.cast['uint8'](img))
    non_zero = np.nonzero(maps[:, :, 1])
    radius = maps[:, :, 1][non_zero]
    cos = maps[:, :, 2][non_zero]
    sin = maps[:, :, 3][non_zero]
    ins.append([non_zero, radius, cos, sin])
    ins.append(np.cast['bool'](maps[:, :, 4]))
    ins.append(cnt)
    ins.append(weight)
    return ins


class MyQueue():
    def __init__(self, maxsize):
        self._q_process = multiprocessing.Queue(maxsize=maxsize)
        self._q_thread = queue.Queue(maxsize=maxsize)

        def steal_daemon():
            print("steal_daemon started.")
            while True:
                self._q_thread.put(self._q_process.get())

        self._steal_thread = threading.Thread()
        self._steal_thread.run = steal_daemon
        self._steal_thread.setDaemon(True)
        self._steal_thread.start()

    def put(self, data, path):
        self._q_process.put(data)
        if path:
            for i in range(len(data[0])):
                pickle.dump(compress(data[0][i], data[1][i, :, :], data[2][i, :, :], data[3], data[4][i, :, :]),
                            gzip.open(os.path.join(path, str(hash(data[0][i])) + str(time.time()) + '.bin.gz'), 'wb'))

    def get(self):
        while True:
            self._q_thread.get()
        # return self._q_thread.get()
