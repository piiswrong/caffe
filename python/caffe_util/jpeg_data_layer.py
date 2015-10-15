import caffe
from .jpeg_pack import JPEGPack
import numpy
import ast
import threading
import random

class JPEGDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.param_ = eval(self.param_str)
        print 'JPEGDataLayer initializing with', self.param_
        self.batch_size_ = self.param_['batch_size']
        self.jpeg_pack_ = JPEGPack(self.param_['source'])
        sample = self.jpeg_pack_.get(self.param_['segment'], 0, self.param_['color'])[0]
        assert len(top) > 0, 'Need at least 1 top blob.'

        self.mirror_ = self.param_.get('mirror', False)
        self.mean_sub_ = self.param_.get('mean_sub', False)
        self.mean_sub2_ = self.param_.get('mean_sub2', False)
        self.scale_ = self.param_.get('scale', 1.0)
        self.scale2_ = self.param_.get('scale2', 1.0)

        if 'crop' in self.param_:
            self.crop_ = True
            self.crop_dim_ = self.param_['crop']
            top[0].reshape(self.batch_size_, sample.shape[2], self.crop_dim_[0], self.crop_dim_[1])
        else:
            self.crop_ = False
            top[0].reshape(self.batch_size_, sample.shape[2], sample.shape[0], sample.shape[1])

        self.buffer = numpy.zeros(top[0].data.shape)
        self.index = 0

        if len(top) >= 2:
            self.output_label = True;
            top[1].reshape(self.batch_size_, 1, 1, 1)
            self.label_buffer = numpy.zeros(top[1].data.shape)

        if 'source2' in self.param_:
            assert len(top) == 3, 'Need 3 top blobs when source2 is set.'
            self.output2_ = True
            self.jpeg_pack2_ = JPEGPack(self.param_['source2'])
            sample2 = self.jpeg_pack2_.get(self.param_['segment2'], 0, self.param_['color2'])[0]

            if self.crop_:
                assert sample.shape[0] >= sample2.shape[0], 'First output need to be bigger than second one when cropping.'
                self.ratio = sample.shape[0]/sample2.shape[0]
                assert self.ratio == sample.shape[1]/sample2.shape[1], 'Aspect ratio need to be the same when cropping.'
                assert sample.shape[0] % sample2.shape[0] == 0, 'Ratio need to be integeral when cropping.'
                assert self.crop_dim_[0] % self.ratio == 0, 'Cropping size need to match when two outputs are given.'
                self.crop_dim2_ = (self.crop_dim_[0]/self.ratio, self.crop_dim_[1]/self.ratio)
                top[2].reshape(self.batch_size_, sample2.shape[2], self.crop_dim2_[0], self.crop_dim2_[1])
            else:
                top[2].reshape(self.batch_size_, sample2.shape[2], sample2.shape[0], sample2.shape[1])
            self.buffer2 = numpy.zeros(top[2].data.shape)
            self.index2 = 0
        else:
            assert len(top) < 3, 'Need less than 3 top blobs when source2 is not set.'
            self.output2_ = False

        self.stop_ = False
        self.worker = threading.Thread(target=self.fetcher)
        self.worker.start()

    def reshape(self, bottom, top):
        pass

    def fetcher(self):
        try:
            for i in xrange(self.batch_size_):
                sample, fname, label = self.jpeg_pack_.get(self.param_['segment'], self.index, self.param_['color'], self.mean_sub_)
                if self.crop_:
                    if self.output2_:
                        cx = random.randint(0, (sample.shape[0] - self.crop_dim_[0])/self.ratio) * self.ratio
                        cy = random.randint(0, (sample.shape[1] - self.crop_dim_[1])/self.ratio) * self.ratio
                    else:
                        cx = random.randint(0, (sample.shape[0] - self.crop_dim_[0]))
                        cy = random.randint(0, (sample.shape[1] - self.crop_dim_[1]))
                    sample = sample[cx:cx+self.crop_dim_[0], cy:cy+self.crop_dim_[1], :]
                if self.mirror_:
                    flag_mirror = random.random() < 0.5
                    if flag_mirror:
                        sample = numpy.fliplr(sample)
                self.buffer[i,...] = sample.transpose((2,0,1)) * self.scale_
                if self.output_label:
                    self.label_buffer[i,0,0,0] = label
                if self.output2_:
                    sample2, fname, label = self.jpeg_pack2_.get(self.param_['segment2'], self.index, self.param_['color2'], self.mean_sub2_)
                    if self.crop_:
                        cx2 = cx / self.ratio
                        cy2 = cy / self.ratio
                        sample2 = sample2[cx2:cx2+self.crop_dim2_[0], cy2:cy2+self.crop_dim2_[1]]
                    if self.mirror_ and flag_mirror:
                        sample2 = numpy.fliplr(sample2)
                    self.buffer2[i,...] = sample2.transpose((2,0,1)) * self.scale2_

                self.index += 1
        except:
            self.worker_succeed = False
            raise
        else:
            self.worker_succeed = True

    def forward(self, bottom, top):
        self.worker.join()
        assert self.worker_succeed, 'Prefetching failed.'
        top[0].data[...] = self.buffer
        if self.output_label:
            top[1].data[...] = self.label_buffer
        if self.output2_:
            top[2].data[...] = self.buffer2
        self.worker = threading.Thread(target=self.fetcher)
        self.worker.start()

    def backward(self, top, propagate_down, bottom):
        pass