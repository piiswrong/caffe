#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, TILE_DIM>
__global__ void ComputeDistanceKernel(Dtype *x, Dtype *y, Dtype *z, int m, int n, int p) {
    __shared__ Dtype sx[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype sy[TILE_DIM][TILE_DIM+1];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int i = bx*TILE_DIM + tx;
    const int j = by*TILE_DIM + ty;

    Dtype c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        sx[ty][tx] = x[j*p + k + tx];
        sy[ty][tx] = y[(k+tx)*n + i];
        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += (sx[ty][kk] - sy[kk][tx])*(sx[ty][kk] - sy[kk][tx]);
        }
        __syncthreads();
    }
    z[i+j*n] = sqrt(c);
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
	dim3 grid(bottom[0]->num()/TILE_DIM, N_/TILE_DIM, 1);
	dim3 block(TILE_DIM, TILE_DIM, 1);
	ComputeDistanceKernel<Dtype, TILE_DIM><<<grid, block>>>(bottom[0]->gpu_data(), 
		this->blobs_[0]->gpu_data(), distance_.gpu_data(), bottom[0]->num(), N_, K_);

	const Dtype *cpu_distance = distance_.cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
	Dtype *cpu_min_distance = min_distance_.mutable_cpu_data();
	Dtype *cpu_label = bottom[1].cpu_data();

	Dtype loss = Dtype(0);

	for (int i = 0; i < bottom[0]->num(); ++i) {
		Dtype min = cpu_distance[i*N_];
		int ind = 0;
		for (int j = 0; j < N_; ++j) {
			if (min > cpu_distance[i*N_ + j]) {
				ind = j
				min = cpu_distance[i*N_ + j];
			}
			cpu_mask[i*N_ + j] = Dtype(0);
		}
		cpu_mask[i*N_ + ind] = Dtype(1);
		cpu_min_distance[i] = min;

		Dtype y = cpu_label[i];
		loss += lambda_ * y * std::min(min, margin_) + (Dtype(1) - lambda_)*(Dtype(1) - y)*std::max(margin_-min, Dtype(0));
	}
	(*top[0])->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  
}

INSTANTIATE_CLASS(CrossLossLayer);

}  // namespace caffe
