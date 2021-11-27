#ifndef MAP_REDUCE_CUH
#define MAP_REDUCE_CUH

// GPU parameters
const int GRID_SIZE = 16;
const int BLOCK_SIZE = 512;

using uint64_cu = unsigned long long int;

const bool SAVE_TO_FILE = true;
const int ITERATIONS = 999;
const uint64_cu NUM_INPUT = 10000;
const int NUM_PAIRS = 1;
const int NUM_OUTPUT = 15;

struct Point{
    uint64_cu x;
    uint64_cu y;

    // 重载<<操作符
    friend std::ostream& operator<<(std::ostream& os, const Point& point){
        os << "Point: (" << point.x << "," << point.y << ")";
        
        return os;
    }
    // 重载+运算符
    // friend const Point operator+(const Point& another){
    //     return Point(this->x + another.x, this->y + another.y);
    // }

    // 重载+=运算符
    // friend Point& operator+=(const Point& another){
    //     this->x += another.x;
    //     this->y += another.y;
    //     return *this;
    // }
};
struct ClusterInfo{
    int cluster_id;
    int cluster_startindex;
    int cluster_len;
};

using input_type = Point;
using output_type = Point;
using key_type = int;
using value_type = Point;
using clusterinfo_type = ClusterInfo;

struct keyvalue_pair{
    key_type key;
    value_type value;

    
    friend std::ostream& operator<<(std::ostream& os, const keyvalue_pair& pair){
        os << "Key: " << pair.key << ", Point: " << pair.value.x << " " << pair.value.y;
        os << std::endl;

        return os;
    }
};

// compare by key
struct keyvaluePairCompare{
    __host__ __device__ bool operator()(const keyvalue_pair& left, const keyvalue_pair& right){
        return left.key < right.key;
    }
};

const uint64_cu TOTAL_PAIRS = NUM_INPUT * NUM_PAIRS;

void runTask(const input_type *input, output_type *output, clusterinfo_type *h_cluster_info);

#endif