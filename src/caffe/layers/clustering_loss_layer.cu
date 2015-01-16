#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void ComputeDistanceKernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
    __shared__ Dtype sx[TILE_DIM][TILE_DIM+1];
    __shared__ Dtype sy[TILE_DIM][TILE_DIM+1];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int i = bx*TILE_DIM + tx;
    const int j = by*TILE_DIM + ty;

    Dtype c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        sx[ty][tx] = x[j*p + k + tx];
        sy[ty][tx] = y[(k+ty)*n + i];
        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += (sx[ty][kk] - sy[kk][tx])*(sx[ty][kk] - sy[kk][tx]);
        }
        __syncthreads();
    }
    z[i+j*n] = sqrt(c/p);
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid(N_/TILE_DIM, bottom[0]->num()/TILE_DIM, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);
  ComputeDistanceKernel<Dtype, ClusteringLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(bottom[0]->gpu_data(),
   this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data(), bottom[0]->num(), N_, K_);

	//std::cout << "distance: " << distance_.asum_data()/distance_.count() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
	Dtype *cpu_min_distance = (*top)[3]->mutable_cpu_data();
	const Dtype *cpu_label = bottom[1]->cpu_data();
  Dtype *cpu_margin = margin_.mutable_cpu_data();
	Dtype *cpu_coef = coef_.mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

	Dtype loss = Dtype(0);
  for (int i = 0; i < N_; i++) cpu_count[i] = Dtype(0);

	for (int i = 0; i < bottom[0]->num(); ++i) {
		Dtype min = cpu_distance[i*N_];
		int ind = 0;
		for (int j = 0; j < N_; ++j) {
			if (min > cpu_distance[i*N_ + j]) {
				ind = j;
				min = cpu_distance[i*N_ + j];
			}
			cpu_mask[i*N_ + j] = Dtype(0);
		}
		cpu_mask[i*N_ + ind] = Dtype(1);
    cpu_ind[i] = Dtype(ind); 
    cpu_count[ind] += Dtype(1);
		cpu_min_distance[i] = min;

		Dtype y = cpu_label[i];
		loss += (lambda_ * y * std::min(min, cpu_margin[ind]) + (Dtype(1) - lambda_)*(Dtype(1) - y)*std::max(cpu_margin[ind]-min, Dtype(0)))/Dtype(2)/bottom[0]->num();

		Dtype sign = min < cpu_margin[ind];
		cpu_coef[i] = (lambda_*y*sign/(min + Dtype(1e-8)) - (Dtype(1) - lambda_)*(Dtype(1) - y)*sign/(min + Dtype(1e-8)))/Dtype(2)/bottom[0]->num();
	}
	(*top)[0]->mutable_cpu_data()[0] = loss;
  Dtype std = Dtype(0);
  for (int i = 0; i < N_; i++) std += cpu_count[i]*cpu_count[i];
  (*top)[1]->mutable_cpu_data()[0] = std/N_ - (Dtype(bottom[0]->num())/N_)*(Dtype(bottom[0]->num())/N_);
	//std::cout << "loss:" << loss << std::endl;
	//std::cout << "coef:" << coef_.asum_data()/coef_.count() << std::endl;
  std::cout << "margin: " << margin_.stat_data() << std::endl;
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/K_;		
  if (propagate_down[0]) {
  	caffe_copy((*bottom)[0]->count(), (*bottom)[0]->gpu_data(), (*bottom)[0]->mutable_gpu_diff());
  	caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, Dtype(-1.0), 
  		mask_.gpu_data(), this->blobs_[0]->gpu_data(), Dtype(1.0), (*bottom)[0]->mutable_gpu_diff());
  	Dtype *gpu_diff = (*bottom)[0]->mutable_gpu_diff();
  	const Dtype *cpu_coef = coef_.cpu_data();	
  	for (int i = 0; i < (*bottom)[0]->num(); ++i) {
  		caffe_gpu_scal(K_, cpu_coef[i]*weight, gpu_diff + i*K_);
  	}
    //LOG(INFO) << "bottom diff:" << (*bottom)[0]->asum_diff()/(*bottom)[0]->count();
  }
  
  if (this->param_propagate_down_[0]) {
    CHECK(propagate_down[0]) << "Only support param_propagate_down when propagate_down is true.";
  	caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), Dtype(Dtype(-1.0)), 
  		(*bottom)[0]->gpu_diff(), mask_.gpu_data(), Dtype(0), this->blobs_[0]->mutable_gpu_diff());
    //LOG(INFO) << "center diff:" << this->blobs_[0]->asum_diff()/this->blobs_[0]->count();
  }

  Dtype *cpu_margin = margin_.mutable_cpu_data();
  const Dtype *cpu_count = count_.cpu_data();
  for (int i = 0; i < N_; i++) {
    cpu_margin[i] += 0.01 * ((Dtype((*bottom)[0]->num())/N_) - cpu_count[i]);
    if (cpu_margin[i] < Dtype(0)) cpu_margin[i] = Dtype(0);
  }
}

INSTANTIATE_CLASS(ClusteringLossLayer);

}  // namespace caffe
