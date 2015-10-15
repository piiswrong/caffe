import cv2
import cv
import caffe
import numpy as np
import ast
import datetime
import time

class ImshowLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) == 0, 'ImshowLayer has no output.'
        self.param_ = ast.literal_eval(self.param_str)
        if 'resize' not in self.param_ or self.param_['resize'] == 0:
            self.resize = False
        else:
            self.resize = True
            self.size = self.param_['resize']
        self.save = self.param_.get('save', None)
        self.scale = self.param_.get('scale', [])
        self.format = self.param_.get('format', [])

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        batch_size = bottom[0].num
        height = 0
        width = 0
        if self.resize:
            width = self.size  * len(bottom)
            height = self.size
        else: 
            for i in xrange(len(bottom)):
                width += bottom[i].width
                height = max(height, bottom[i].height)
        buff = np.zeros((height*batch_size, width, 3), dtype = np.uint8)
        #import pdb 
        #pdb.set_trace()
        for i in xrange(batch_size):
            cur = 0
            for j in xrange(len(bottom)):
                img = bottom[j].data[i].transpose((1,2,0))
                if len(self.scale):
                    assert len(self.scale) == len(bottom)
                    img *=  self.scale[j]
                img = img.astype(np.uint8)
                if len(self.format):
                    assert len(self.format) == len(bottom)
                    if self.format[j] == 'ycrcb':
                        img = cv2.cvtColor(img, cv.CV_YCrCb2BGR)
                if img.shape[2] == 1:
                    img = np.tile(img, 3)
                if self.resize:
                    widthj = heightj = self.size
                    img = cv2.resize(img, (self.size, self.size))
                else:
                    widthj = bottom[j].width
                    heightj = bottom[j].height
                buff[i*height:i*height+heightj, cur:cur+widthj, :]  = img
                cur += widthj
        if self.save is None:
            cv2.imshow('buff', buff)
            cv2.waitKey(0)
        else:
            cv2.imwrite(self.save+'%f'%time.time()+'.jpg', buff)

    def backward(self, top, propagate_down, bottom):
        pass