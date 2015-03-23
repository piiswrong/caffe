#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/iostreams/device/mapped_file.hpp>
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
JPEGDataLayer<Dtype>::~JPEGDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  source->close();
  delete source;
  delete cv_img;
}

template <typename Dtype>
void JPEGDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize mmap
  FILE *fin = fopen(this->layer_param_.jpeg_data_param().info().c_str(), "r");
  CHECK(fin) << "cannot open info file " << this->layer_param_.jpeg_data_param().info();
  int offset, size, label;
  while (EOF != fscanf(fin, "%*s\t%d\t%d\t%d\n", &offset, &size, &label)) {
    offset_.push_back(offset);
    size_.push_back(size);
    label_.push_back(label);
  }
  fclose(fin);

  source = new boost::iostreams::mapped_file_source(boost::iostreams::mapped_file_params(this->layer_param_.jpeg_data_param().source()));
  CHECK(source->is_open()) << "cannot open source " << this->layer_param_.jpeg_data_param().source();
  cursor_ = 0;

  cv_img = new cv::Mat;

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(DecodeImageToDatum(source->data(), size_[0], *cv_img, 0, 0, 0, this->layer_param_.jpeg_data_param().is_color(), &datum)) << "cannot read from source!";
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.jpeg_data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.jpeg_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.jpeg_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.jpeg_data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void JPEGDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.jpeg_data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(DecodeImageToDatum(source->data()+offset_[cursor_], size_[cursor_], *cv_img, label_[cursor_], 0, 0, this->layer_param_.jpeg_data_param().is_color(), &datum)) << "cannot read from source!";
    cursor_ = (cursor_+1)%offset_.size();

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
  }
}

INSTANTIATE_CLASS(JPEGDataLayer);

}  // namespace caffe
