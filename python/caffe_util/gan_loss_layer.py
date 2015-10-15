import caffe
import numpy as np

class GANLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) >= 1 and len(top) <= 2
        assert len(bottom) == 2
        self.param = eval(self.param_str)
        self.phases = self.param['phases']

    def reshape(self, bottom, top):
        top[0].reshape(1,1,1,1)
        if len(top) > 1:
            top[1].reshape(1,2,1,1)

    def forward(self, bottom, top):
        x = bottom[0].data.squeeze()
        y = bottom[1].data.squeeze()
        top[0].data.flat[0] = np.mean(np.log(1.0+np.exp(-np.abs(x))) - x*(y-(x>0)))
        if len(top) > 1:
            top[1].data.flat[0] = 1.0 - np.mean(np.abs(y - (x>0)))
            top[1].data.flat[1] = -np.mean((1.0-y)*np.log(1.0/(1.0+np.exp(-x))))*2.0

    def backward(self, top, propagate_down, bottom):
        assert not propagate_down[1]
        if not propagate_down[0]:
            return
        pos = caffe.iter()%len(self.phases)
        prob = 1.0/(1.0 + np.exp(-bottom[0].data.squeeze()))
        if self.phases[pos]:
            bottom[0].diff.flat[...] = (top[0].diff.squeeze()*(1.0-bottom[1].data.squeeze())*(prob - 1.0)/bottom[0].num*2.0).flat
        else:
            bottom[0].diff.flat[...] = (top[0].diff.squeeze()*(prob - bottom[1].data.squeeze())/bottom[0].num).flat