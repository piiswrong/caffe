#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype, int TILE_DIM>
__global__ void distance_kernel(const Dtype *x, const Dtype *y, Dtype *z, int m, int n, int p) {
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

/*
template<typename Dtype>
__global__ void coef_kernel(int m, int n, const Dtype *dist, const Dtype *y, const Dtype *margin, Dtype *coef, Dtype *loss, Dtype lambda) {
  CUDA_KERNEL_LOOP(index, m*n) {
    int i = index/n;
    int j = index%n;
    Dtype d = dist[index]/margin[j];
    Dtype sign = d < Dtype(1);
    coef[index] = ( lambda * y[i] - (Dtype(1)-lambda)*(Dtype(1)-y) ) * sign / (margin[j]*dist[index]*m*n);
    loss[index] = ( lambda * y[i] * min(d, Dtype(1)) + (Dtype(1)-lambda)*(Dtype(1)-y[i])*max(Dtype(1)-d, Dtype(0)) )/Dtype(m*n);
  } 
}
*/

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  dim3 grid(N_/TILE_DIM, bottom[0]->num()/TILE_DIM, 1);
  dim3 block(TILE_DIM, TILE_DIM, 1);
  distance_kernel<Dtype, ClusteringLossLayer<Dtype>::TILE_DIM><<<grid, block>>>(bottom[0]->gpu_data(),
   this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data(), bottom[0]->num(), N_, K_);

	//std::cout << "distance: " << distance_.asum_data()/distance_.count() << std::endl;

	const Dtype *cpu_distance = distance_.cpu_data();
	Dtype *cpu_mask = mask_.mutable_cpu_data();
	Dtype *cpu_min_distance = (*top)[3]->mutable_cpu_data();
	const Dtype *cpu_label = bottom[1]->cpu_data();
  Dtype *cpu_margin = this->blobs_[1]->mutable_cpu_data();
	Dtype *cpu_coefm = coefm_.mutable_cpu_data();
  Dtype *cpu_coefn = coefn_.mutable_cpu_data();
  Dtype *cpu_ind = (*top)[2]->mutable_cpu_data();
  Dtype *cpu_count = count_.mutable_cpu_data();

	Dtype loss = Dtype(0);
  for (int i = 0; i < N_; i++) {
    cpu_count[i] = Dtype(0);
    cpu_coefn[i] = Dtype(0);
  }

	for (int i = 0; i < bottom[0]->num(); ++i) {
    Dtype sum = Dtype(0);
		int ind = 0;
    Dtype y = cpu_label[i];
    cpu_coefm[i] = Dtype(0);
		for (int j = 0; j < N_; ++j) {
      Dtype sign = Dtype(0);
      Dtype d = cpu_distance[i*N_ + j]/cpu_margin[j];
			if (d < Dtype(1)) {
				sign = Dtype(1);
        cpu_count[j] += (Dtype(1)-y) * sign * (Dtype(1)-d);
			}
			cpu_mask[i*N_ + j] = (lambda_*y - (Dtype(1)-lambda_)*(Dtype(1)-y))*sign/((cpu_distance[i*N_ + j] + Dtype(1e-10))*cpu_margin[ind]*bottom[0]->num());
      cpu_coefm[i] += cpu_mask[i*N_ + j];
      cpu_coefn[j] += cpu_mask[i*N_ + j];
      loss += (lambda_ * y * std::min(d, Dtype(1)) + (Dtype(1)-lambda_)*(Dtype(1)-y)*std::max(Dtype(1)-d, Dtype(0)))/bottom[0]->num();
		}
  }
	(*top)[0]->mutable_cpu_data()[0] = loss;
  Dtype std = Dtype(0);
  Dtype mean = Dtype(0);
  for (int i = 0; i < N_; i++) { 
    std += cpu_count[i]*cpu_count[i];
    mean += cpu_count[i];
  }
  std /= N_;
  mean /= N_;
  (*top)[1]->mutable_cpu_data()[0] = std::sqrt(std - mean*mean)/mean;
  std::cout << "margin: " << this->blobs_[1]->stat_data() << std::endl;
  std::cout << "count: " << count_.stat_data() << std::endl;
  std::cout << "dist: " << distance_.stat_data() << std::endl;
  std::cout << "mask: " << mask_.stat_data() << std::endl;
  std::cout << "coefn: " << coefn_.stat_data() << std::endl;
}

template <typename Dtype>
void ClusteringLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype weight = top[0]->cpu_diff()[0]/K_;		
  if (propagate_down[0]) {
    caffe_gpu_dgmm(CblasLeft, (*bottom)[0]->num(), K_, (*bottom)[0]->gpu_data(),
      coefm_.gpu_data(), (*bottom)[0]->mutable_gpu_diff());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, mask_.num(), K_, N_, -weight, 
      mask_.gpu_data(), this->blobs_[0]->gpu_data(), weight, (*bottom)[0]->mutable_gpu_diff());
    //LOG(INFO) << "bottom diff:" << (*bottom)[0]->asum_diff()/(*bottom)[0]->count();
  }
  
  if (this->param_propagate_down_[0]) {
    caffe_gpu_dgmm(CblasRight, K_, N_, this->blobs_[0]->gpu_data(), coefn_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
  	caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, mask_.num(), -weight, 
  		(*bottom)[0]->gpu_data(), mask_.gpu_data(), weight, this->blobs_[0]->mutable_gpu_diff());
    //std::cout << "diff: " <<  this->blobs_[0]->stat_diff() << std::endl;
    //LOG(INFO) << "center diff:" << this->blobs_[0]->asum_diff()/this->blobs_[0]->count();
  }
  
  Dtype *cpu_margin = this->blobs_[1]->mutable_cpu_data();
  const Dtype *cpu_count = count_.cpu_data();
  const Dtype beta = this->layer_param_.clustering_loss_param().beta();
  for (int i = 0; i < N_; i++) {
    cpu_margin[i] -= 0.1 * (cpu_count[i] - beta);
    if (cpu_margin[i] < Dtype(1e-4)) cpu_margin[i] = Dtype(1e-4);
  }
}

INSTANTIATE_CLASS(ClusteringLossLayer);

}  // namespace caffe
