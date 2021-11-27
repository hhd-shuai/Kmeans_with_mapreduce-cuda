#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <exception>

#include "config.cuh"
#include "random_num_generator.hpp"

// input_type, output_type -> Point
void initialize_centroids(input_type *input, output_type *output){
    RandomNumGenerator distribution(NUM_INPUT);
    // NUM_OUTPUT = 15, 随机生成15个质心
    for(int i = 0; i < NUM_OUTPUT; ++i){
        int random_num = distribution.sample();
        output[i] = input[random_num];
    }
}


// 1. 必须使用MapReduce的概念
// 2. 必须关于GPU方面的MapReduce实现
// 3. 必须使用CUDA或OPENCL
// 4. Loading Balancing
int main(int argc, char *argv[]){
    if(argc != 2){
        std::cout << "input textfile *.txt" << std::endl;
        exit(1);
    }
    using mlsceond = std::chrono::milliseconds;
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    // 数据加载计时
    auto read_data_start = steady_clock::now();
    // 获取文件名
    std::string fname = argv[1];
    //host端分配内存
    size_t input_size = NUM_INPUT * sizeof(input_type);
    input_type *input = (input_type *)malloc(input_size);
    size_t output_size = NUM_OUTPUT * sizeof(output_type);
    output_type *output = (output_type *)malloc(output_size);
    size_t cluster_info_size = NUM_OUTPUT * sizeof(clusterinfo_type);
    clusterinfo_type *h_cluster_info = (clusterinfo_type *)malloc(cluster_info_size);;

    // 测试
    // size_t cluster_info_size = NUM_OUTPUT * sizeof(clusterinfo_type);
    // clusterinfo_type *clusterinfo_output = (clusterinfo_type *)malloc(cluster_info_size);

    // 读取数据
    std::string line;
    std::ifstream input_file(fname);
    if(!input_file.is_open()){
        std::cout << "File could not be opened." << std::endl;
        exit(1);
    }
    try{
        for(size_t index = 0; index < NUM_INPUT; ++index){
            getline(input_file, line);
            std::istringstream buffer(line);

            buffer >> input[index].x;
            buffer >> input[index].y;
        }
        input_file.close();
    }catch(std::exception& e){
        std::cout << "Failed to read file." << std::endl;
        exit(1);
    }

    // for(int i = 0; i < 10; ++i){
    //     std::cout << input[i] << "\t";
    // }
    initialize_centroids(input, output);
    auto read_data_end = steady_clock::now();
    
    // 执行mapreduce task
    runTask(input, output, h_cluster_info);
    // runTask(input, output, clusterinfo_output);

    // 测试
    // for(size_t i = 0; i < NUM_OUTPUT; ++i){
    //     std::cout << clusterinfo_output[i].cluster_id << " ";
    //     std::cout << clusterinfo_output[i].cluster_startindex  << " ";
    //     std::cout << clusterinfo_output[i].cluster_len << " ";
    //     std::cout << std::endl;     
    // }
    /*0 0 120 
      1 120 365 
      2 485 42 
      3 527 278 
      4 805 705 
      5 1510 289 
      6 1799 151 
      7 1950 437 
      8 2387 434 
      9 2821 461 
      10 3282 693 
      11 3975 308 
      12 4283 119 
      13 4402 495 
      14 4897 103 */
      
    // 保存文件 
    std::ofstream output_file;
    if(SAVE_TO_FILE){
        std::string out_fname = fname + ".output";
        output_file.open(out_fname);
        if(!output_file.is_open()){
            std::cout << "File open failed." << std::endl;
            exit(1);
        }
    }
    // 打印输出
    for(size_t i = 0; i < NUM_OUTPUT; ++i){
        std::cout << output[i] << std::endl;
        if(SAVE_TO_FILE){
            output_file << output[i] << std::endl;
        }
    }

    auto task_time_end = steady_clock::now();

    free(input);
    free(output);
    free(h_cluster_info);

    auto total_time_end = steady_clock::now();

    auto time1 = duration_cast<mlsceond>( read_data_end - read_data_start ).count();
    auto time2 = duration_cast<mlsceond>( task_time_end - read_data_end ).count();
    auto total_time = duration_cast<mlsceond>( total_time_end - read_data_start ).count();

    std::cout << "Data loading and initialize: " << time1 << " milliseconds" << std::endl;
    std::cout << "Time for map reduce KMeans: " << time2 << " milliseconds" << std::endl;
    std::cout << "Total time: " << total_time << " milliseconds" << std::endl;

    return 0;
}