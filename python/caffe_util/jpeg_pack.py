import cv2
import cv
import mmap
import os
import random
import numpy as np
import errno
import caffe

class JPEGPack(object):
    """
    JPEGPack is an interface for creating and reading binaries of condensed
    jpeg images.
    """
    def __init__(self, path, create_mode = False):
        self.path_ = path
        self.segments_ = []
        if create_mode:
            self.mm_ = None
            self.create_mode_ = True
        else:
            self.create_mode_ = False
            fpack = open(self.path_, "rb")
            self.mm_ = mmap.mmap(fpack.fileno(), 0, prot=mmap.PROT_READ)
            i = 0
            while True:
                fname = self.path_ + "." + str(i)
                if os.path.isfile(fname):
                    finfo = open(fname, "r")
                    self.segments_.append([])
                    for line in finfo.readlines():
                        line = line.strip().split("\t")
                        assert len(line) == 4
                        self.segments_[-1].append((line[0], int(line[1]), int(line[2]), int(line[3])))
                else:
                    break
                i += 1
            if os.path.isfile(self.path_+"_mean_color.npy"):
                self.mean_color_ = np.load(self.path_+"_mean_color.npy")
            else:
                self.mean_color_ = None
            if os.path.isfile(self.path_+"_mean_grey.npy"):
                self.mean_grey_ = np.load(self.path_+"_mean_grey.npy")
            else:
                self.mean_grey_ = None
            if os.path.isfile(self.path_+"_mean_ycrcb.npy"):
                self.mean_ycrcb_ = np.load(self.path_+"_mean_ycrcb.npy")
            else:
                self.mean_ycrcb_ = None



    def __del__(self):
        if self.mm_:
            self.mm_.close()

    def num_segment(self):
        return len(self.segments_)

    def num_entry(self, segment):
        return len(self.segments_[segment])

    def write_info(self):
        for i in xrange(len(self.segments_)):
            with open(self.path_+"."+str(i), 'w') as fout:
                for line in self.segments_[i]:
                    fout.write('%s\t%d\t%d\t%d\n'%line)

    def compute_mean(self, color = True):
        assert not self.create_mode_, 'compute_mean should not be called in create mode.'

        mean = None
        for i in xrange(self.num_entry(0)):
            img, fname, label = self.get(0, i, color)
            if mean is None:
                mean = np.zeros(img.shape, dtype=np.float32)
            if mean.shape != img.shape:
                print 'Cannot compute mean on non-uniform sized images.'
                return
            mean += img
        mean /= self.num_entry(0)
        if color == True:
            self.mean_color_ = mean
            np.save(self.path_+"_mean_color.npy", mean)
        elif color == False:
            self.mean_grey_ = mean
            np.save(self.path_+"_mean_grey.npy", mean)
        elif color == 'ycrcb':
            self.mean_ycrcb_ = mean
            np.save(self.path_+"_mean_ycrcb.npy", mean)



    def create(self, image_root, segments = (1, 1), recursive = False):
        assert self.create_mode_, 'create should be called in create mode.'

        image_list = []
        if recursive:
            cat = {}
            for path, subdirs, files in os.walk(image_root):
                for fname in files:
                    fpath = os.path.join(path, fname)
                    suffix = os.path.splitext(fname)[1].lower()
                    if os.path.isfile(fpath) and (suffix == ".jpeg" or suffix == '.jpg'):
                        if path not in cat:
                            cat[path] = len(cat)
                        image_list.append((fpath, os.path.getsize(fpath), cat[path]))
        else:
            for fname in os.listdir(image_root):
                fpath = os.path.join(image_root, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix == ".jpeg" or suffix == '.jpg'):
                    image_list.append((fpath, os.path.getsize(fpath), 0))

        N = len(image_list)
        assert N > 0, 'No image found in path ' + image_root
        print 'total # of images', N
        p = np.random.permutation(N)
        inverse_p = np.argsort(p)
        cum_size = [0]
        for i in xrange(1, N):
            cum_size.append(cum_size[i-1] + image_list[p[i-1]][1])
        total_size = cum_size[-1] + image_list[p[-1]][1]

        seg_size = [ int(float(segments[i])/sum(segments)*N) for i in xrange(len(segments)-1) ]
        seg_size.append(N - sum(seg_size))
        base = 0
        for i in xrange(len(segments)):
            self.segments_.append([])
            with open(self.path_ + "." + str(i), 'w') as finfo:
                for j in xrange(seg_size[i]):
                    info = (image_list[p[base + j]][0],cum_size[base + j],image_list[p[base + j]][1],image_list[p[base + j]][2])
                    finfo.write('%s\t%d\t%d\t%d\n'%info)
                    self.segments_[-1].append(info)
            base += seg_size[i]


        with open(self.path_, 'wb') as fout:
            fout.write('\0')

        fout = open(self.path_, "r+b")
        self.mm_ = mmap.mmap(fout.fileno(), 0)
        self.mm_.resize(total_size)

        for i in xrange(N):
            if i%1000 == 0:
                print 'creating', i*1.0/N
            with open(image_list[i][0], 'rb') as fin:
                start = cum_size[inverse_p[i]]
                end = start + image_list[i][1]
                self.mm_[start:end] = fin.read()

        self.mm_.flush()
        self.create_mode_ = False
        self.compute_mean(color = True)
        self.compute_mean(color = False)
        self.compute_mean(color = 'ycrcb')


    def get(self, segment, index, color = True, mean_sub = False):
        assert not self.create_mode_, 'get should not be called in create mode.'
        if index >= len(self.segments_[segment]):
            index = index % len(self.segments_[segment])
        start = self.segments_[segment][index][1]
        size = self.segments_[segment][index][2]
        data = np.fromstring(self.mm_[start:(start+size)], dtype=np.uint8)
        if color == True:
            img = cv2.imdecode(data, cv.CV_LOAD_IMAGE_COLOR).astype(np.float32)
            if mean_sub:
                img -= self.mean_color_
        elif color == False:
            img = cv2.imdecode(data, cv.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32)
            img = img[:,:,np.newaxis]
            if mean_sub:
                img -= self.mean_grey_
        elif color == 'ycrcb':
            img = cv2.imdecode(data, cv.CV_LOAD_IMAGE_COLOR)
            img = cv2.cvtColor(img, cv.CV_BGR2YCrCb).astype(np.float32)
            if mean_sub:
                img -= self.mean_ycrcb_
        elif color == 'crcb':
            img = cv2.imdecode(data, cv.CV_LOAD_IMAGE_COLOR)
            img = cv2.cvtColor(img, cv.CV_BGR2YCrCb).astype(np.float32)
            if mean_sub:
                img -= self.mean_ycrcb_
            img = img[:,:,1:]
        return img, self.segments_[segment][index][0], self.segments_[segment][index][3]
        
    def create_transform(self, source_pack, mapping, color=True):
        assert self.create_mode_, 'create_transform should be called in create mode.'

        with open(self.path_, 'wb') as fout:
            fout.write('\0')
        fout = open(self.path_, "r+b")
        self.mm_ = mmap.mmap(fout.fileno(), 0)
        cur = 0
        for i in xrange(source_pack.num_segment()):
            self.segments_.append([])
            for j in xrange(source_pack.num_entry(i)):
                if j % 1000 == 0:
                    print 'transforming', i, j*1.0/source_pack.num_entry(i)
                img, fname, label = source_pack.get(i,j,color)
                img = mapping(img)
                if img is None:
                    continue
                ret, buf = cv2.imencode('.JPEG', img)
                assert ret
                buf = buf.tostring()
                self.mm_.resize(cur + len(buf))
                self.mm_.write(buf)
                self.segments_[-1].append((fname, cur, len(buf), label))
                cur += len(buf)

        self.write_info()
        self.mm_.flush()
        self.create_mode_ = False
        self.compute_mean(color = True)
        self.compute_mean(color = False)
        self.compute_mean(color = 'ycrcb')


