#include <random>
#include <chrono>
#include <stdio.h>

// 根据输入参数max，随机生成0-max int型随机数
class RandomNumGenerator
{
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
public:
    RandomNumGenerator(int max);
    ~RandomNumGenerator();
    int sample();
};

RandomNumGenerator::RandomNumGenerator(int max)
    : generator(std::chrono::system_clock::now().time_since_epoch().count()), distribution(0,max)
{
}

RandomNumGenerator::~RandomNumGenerator()
{
}

int RandomNumGenerator::sample(){
    return distribution(generator);
}