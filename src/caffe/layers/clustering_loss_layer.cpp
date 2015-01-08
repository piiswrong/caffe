#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ClusteringLossLayer<Dtype>::LayerSetup(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetup(bottom, top);
  CHECK_EQ(bottom[0]->num()%TILE_DIM, 0) << "Only support" 
    "batch sizes that are multiples of " << TILE_DIM << ".";
  N_ = this->layer_param_.clustering_loss_param().num_center();
  CHECK_EQ(N_%TILE_DIM, 0) << "Only support" 
    "center numbers that are multiples of " << TILE_DIM << ".";
  K_ = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(K_%TILE_DIM, 0) << "Only support" 
    "input dimensions that are multiples of " << TILE_DIM << ".";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.clustering_loss_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());


  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->count() / bottom[0]->num(), K_) << "Input size "
    "incompatible with inner product parameters.";

  distance_.Reshape(bottom[0]->num(), 1, 1, N_);
  mask_.Reshape(bottom[0]->num(), 1, 1, N_);
  min_distance_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(CrossLossLayer);
#endif

INSTANTIATE_CLASS(CrossLossLayer);

}  // namespace caffe
