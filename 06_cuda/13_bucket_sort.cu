#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int *bucket, int bucket_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= bucket_size) return;
    bucket[i] = 0;
}

__global__ void bucket_reduction(int *bucket, int *key, int key_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= key_size) return;
    atomicAdd(&bucket[key[i]],1);
}

__global__ void bucket_scan(int *bucket, int *a, int bucket_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= bucket_size) return;
    for(int j=1; j<bucket_size; j<<=1){
        a[i] = bucket[i];
        __syncthreads();
        if(i>=j) bucket[i] += a[i-j];
        __syncthreads();
    }
}

__global__ void bucket_sort(int *bucket, int *key, int bucket_size, int key_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= key_size) return;
    key[i] = 0;
    for(int j=1; j<bucket_size; j++){
        if(i >= bucket[j-1] && i < bucket[j]){
            key[i] = j;
            return;
        }
    }
}

int main() {
  int n = 50;
  int range = 5;
  const int M = 1024;
  int *key, *bucket, *a;

  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&a, range*sizeof(int));
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_init<<<(range+M-1)/M,range>>>(bucket, range);
  cudaDeviceSynchronize();

  bucket_reduction<<<(n+M-1)/M,n>>>(bucket, key, n);
  cudaDeviceSynchronize();

  bucket_scan<<<(range+M-1)/M,range>>>(bucket, a, range);
  cudaDeviceSynchronize();
  
  bucket_sort<<<(n+M-1)/M,n>>>(bucket, key, range, n);
  cudaDeviceSynchronize();
  
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
