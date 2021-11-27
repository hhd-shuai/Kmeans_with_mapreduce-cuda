#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "config.cuh"


__device__ __host__
uint64_cu distance(const Point& p1, const Point& p2){
    uint64_cu dist = 0;
    uint64_cu x_dim = p1.x - p2.x;
    uint64_cu y_dim = p1.y - p2.y;

    dist += x_dim * x_dim + y_dim * y_dim;

    return dist;
}

// 计算当前点与k个质心的距离，根据最小的距离将current point划分到对应的簇。
// 生成key-value结构pairs。key->cluster_id value->point
__device__ void mapper(const input_type *d_input, keyvalue_pair *d_pairs, output_type *d_output){
    uint64_cu min_dist = ULLONG_MAX;
    int cluster_id = -1;

    for(int i = 0; i < NUM_OUTPUT; ++i){
        uint64_cu dist = distance(*d_input, d_output[i]);
        if(dist < min_dist){
            min_dist = dist;
            cluster_id = i;
        }
    }

    d_pairs->key = cluster_id;
    d_pairs->value = *d_input;
}

__global__ void mapKernel(const input_type *d_input, keyvalue_pair *d_pairs, output_type *d_output){
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // 防止线程，读到相同数据
    for(size_t i = threadId; i < NUM_INPUT; i += stride){
        mapper(&d_input[i], &d_pairs[i * NUM_PAIRS], d_output);
    }
}

void runMapper(const input_type *d_input, keyvalue_pair *d_pairs, output_type *d_output){
    mapKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_pairs, d_output);
    cudaDeviceSynchronize();
}


// 累加簇中所有点的x,y值。
// 根据累和结果和点数，求得簇中所有点的平均值作为新的质心。
__device__ void reducer(keyvalue_pair *d_pairs, size_t len, output_type *d_output){
    Point new_centroid;
    new_centroid.x = 0;
    new_centroid.y = 0;

    for(size_t i = 0; i < len; ++i){
        new_centroid.x += d_pairs[i].value.x;
        new_centroid.y += d_pairs[i].value.y;
    }

    int cluster_id = d_pairs[0].key;
    new_centroid.x /= len;
    new_centroid.y /= len;

    d_output[cluster_id] = new_centroid;
}

__global__ void reducerKernel(keyvalue_pair *d_pairs, output_type *d_output, clusterinfo_type *cluster_info){
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = threadId; i < NUM_OUTPUT; i += stride){
        size_t start_index = 0;
        size_t end_index = TOTAL_PAIRS;
        size_t cluster_len = 0;
        size_t uniq_key_index = 0;  
        
        for (size_t j=1; j<TOTAL_PAIRS; j++) {
            if (keyvaluePairCompare()(d_pairs[j-1], d_pairs[j])) {
                if (uniq_key_index == i) {
                    end_index = j;
                    break;
                }
                else {
                    uniq_key_index++;
                    start_index = j;
                }
            }
        }

        if (uniq_key_index != i) {
            return; 
        }

        cluster_len = end_index - start_index;

        // reducer(&d_pairs[start_index], cluster_len, &d_output[i]);
        // 此处为了实现loading balance， 先将每个簇在d_pairs中对应的起始start_index， 和cluster_len做记录。
        cluster_info[threadId].cluster_id = d_pairs[start_index].key;
        cluster_info[threadId].cluster_startindex = start_index;
        cluster_info[threadId].cluster_len = cluster_len;
    }
    __syncthreads();  
}
__global__ void reducer1_loading_balance(keyvalue_pair *d_pairs, int len, int index,output_type *d_output){

    __shared__ unsigned long long s_data_x[256];
    __shared__ unsigned long long s_data_y[256];

    int thread_global_id = blockDim.x * blockIdx.x + threadIdx.x;
    // block块内id
    int tid = threadIdx.x;
    
    s_data_x[tid] = 0;
    s_data_y[tid] = 0;
   
    if(thread_global_id < len){
        s_data_x[tid] = d_pairs[thread_global_id].value.x;
        s_data_y[tid] = d_pairs[thread_global_id].value.y;

        __syncthreads();
        // 对簇内点坐标进行规约求和
        for(size_t i = blockDim.x / 2; i > 0; i >>= 1){
            if(tid < i){
                s_data_x[tid] += s_data_x[tid + i];
                s_data_y[tid] += s_data_y[tid + i];
            }
            __syncthreads();
        }
        // 汇总结果
        if(tid == 0){
            d_output[index].x = s_data_x[0] / len;
            d_output[index].y = s_data_y[0] / len;
        }
    }
}
__global__ void reducer2_loading_balance(keyvalue_pair *d_pairs, int len, int index,output_type *d_output){

    __shared__ unsigned long long s_data_x[256];
    __shared__ unsigned long long s_data_y[256];
    // 注意此处全局thread id 计算时要乘以2
    int thread_global_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    // block块内id
    int tid = threadIdx.x;
    
    s_data_x[tid] = 0;
    s_data_y[tid] = 0;
    // 每次从全局内存中读取两个数据，并将两个数据做一次求和，然后写入共享内存，提高由全局内存向共享内存传输数据的带宽利用率，减少数据传输时间。
    if(thread_global_id < len){
        // 
        s_data_x[tid] = d_pairs[thread_global_id].value.x + d_pairs[thread_global_id + blockDim.x].value.x;
        s_data_y[tid] = d_pairs[thread_global_id].value.y + d_pairs[thread_global_id + blockDim.x].value.y;

        __syncthreads();
        // 对簇内点坐标进行规约求和
        for(size_t i = blockDim.x / 2; i > 32; i /= 2){
            if(tid < i){
                s_data_x[tid] += s_data_x[tid + i];
                s_data_y[tid] += s_data_y[tid + i];
            }
            __syncthreads();
        }
        // 每一个块中的第一个warp中的前32个线程完全进行循环展开，进而每个block中都可以有32个线程去掉循环，进而提升算法效率。
        if(tid < 32){
            s_data_x[tid] += s_data_x[tid + 32];
            s_data_y[tid] += s_data_y[tid + 32];
            s_data_x[tid] += s_data_x[tid + 16];
            s_data_y[tid] += s_data_y[tid + 16];
            s_data_x[tid] += s_data_x[tid + 8];
            s_data_y[tid] += s_data_y[tid + 8];
            s_data_x[tid] += s_data_x[tid + 4];
            s_data_y[tid] += s_data_y[tid + 4];
            s_data_x[tid] += s_data_x[tid + 2];
            s_data_y[tid] += s_data_y[tid + 2];
            s_data_x[tid] += s_data_x[tid + 1];
            s_data_y[tid] += s_data_y[tid + 1];
        }
        __syncthreads();
        // 汇总结果
        if(tid == 0){
            d_output[index].x = s_data_x[0] / len;
            d_output[index].y = s_data_y[0] / len;
        }
    }
}
void runReducer(keyvalue_pair *d_pairs, output_type *d_output, clusterinfo_type *cluster_info) {
    reducerKernel<<<1, BLOCK_SIZE>>>(d_pairs, d_output, cluster_info);
    cudaDeviceSynchronize(); 
}
// loading balance
// 根据每个簇的数据量动态分配线程。执行规约操作，以获取新的质心。
void runReducerJob(keyvalue_pair *d_pairs, output_type *d_output, clusterinfo_type *h_cluster_info){
    for(size_t i = 0; i < NUM_OUTPUT; ++i){
        // *******************************
       int current_len = h_cluster_info[i].cluster_len;
       int current_start_index = h_cluster_info[i].cluster_startindex;
       int blockSize = 256;
       int gridSize = (current_len + blockSize - 1) / blockSize;

       //std::cout << "current_len: " << current_len << " current_start_index: " << current_start_index << std::endl;
       //reducer2_loading_balance<<<gridSize, blockSize>>>(&d_pairs[current_start_index], current_len, d_output);
       reducer1_loading_balance<<<gridSize, blockSize>>>(&d_pairs[current_start_index], current_len, i, d_output);
   }
}

// Mapreduce pipline
void runTask(const input_type *input, output_type *output, clusterinfo_type *h_cluster_info){
    cudaError_t erro = cudaSuccess;
    input_type *d_input;
    keyvalue_pair *d_pairs;
    output_type *d_output;
    clusterinfo_type *cluster_info;

    // 分配device端内存
    size_t input_size = NUM_INPUT * sizeof(input_type);
    size_t pair_size = TOTAL_PAIRS * sizeof(keyvalue_pair);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    size_t cluster_info_size = NUM_OUTPUT * sizeof(clusterinfo_type);
    
    erro = cudaMalloc(&d_input, input_size);
    if(erro != cudaSuccess) {std::cout << "allocated memory fault in GPU" << std::endl; return;}
    erro = cudaMalloc(&d_pairs, pair_size);
    if(erro != cudaSuccess) {std::cout << "allocated memory fault in GPU" << std::endl; return;}
    erro = cudaMalloc(&d_output, output_size);
    if(erro != cudaSuccess) {std::cout << "allocated memory fault in GPU" << std::endl; return;}
    erro = cudaMalloc(&cluster_info, cluster_info_size);
    if(erro != cudaSuccess) {std::cout << "allocated memory fault in GPU" << std::endl; return;}

    erro = cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    if(erro != cudaSuccess) {std::cout << "copy data from CPU to GPU fault" << std::endl; return;}
    erro = cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice);
    if(erro != cudaSuccess) {std::cout << "copy data from CPU to GPU fault" << std::endl; return;}

    for(int iter = 0; iter < ITERATIONS; ++iter){
        runMapper(d_input, d_pairs, d_output);
        // Trust为并行算法库
        // 编程接口  thrust::sort(const thrust::detail::execution_policy_base< DerivedPolicy > & 	exec,
        //                  RandomAccessIterator 	first,
        //                  RandomAccessIterator 	last 
        //                  )	
        // Parameters
        // exec	The execution policy to use for parallelization.(thrust::host , thrust::device)
        // first(RandomAccessIterator) The beginning of the sequence.
        // last(RandomAccessIterator) The end of the sequence.
        thrust::sort(thrust::device, d_pairs, d_pairs + TOTAL_PAIRS, keyvaluePairCompare());
        runReducer(d_pairs, d_output, cluster_info);
        erro = cudaMemcpy(h_cluster_info, cluster_info, cluster_info_size, cudaMemcpyDeviceToHost);
        if(erro != cudaSuccess) {std::cout << "copy data from GPU to CPU fault" << std::endl; return;}
        runReducerJob(d_pairs, d_output, h_cluster_info);
        // 测试
        // if(iter == ITERATIONS - 1){
        //     for(int i = 0; i < NUM_OUTPUT; ++i){
        //         std::cout << "h_cluster_info: " << h_cluster_info[i].cluster_id << " " << h_cluster_info[i].cluster_startindex << " " << h_cluster_info[i].cluster_len << std::endl;
        //     }
        // }
    }

    erro = cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    // 测试
    // cudaMemcpy(clusterinfo_output, cluster_info, cluster_info_size, cudaMemcpyDeviceToHost);
    if(erro != cudaSuccess) {std::cout << "copy data from GPU to CPU fault" << std::endl; return;}

    cudaFree(d_input);
    cudaFree(d_pairs);
    cudaFree(d_output);
    cudaFree(cluster_info);
}