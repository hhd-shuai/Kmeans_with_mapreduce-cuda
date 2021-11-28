# Kmeans_with_mapreduce-cuda
#### Heterogeneous Computing
#### Languages: CUDA or OPENCL
MapReduce on GPU
- 必须使用MapReduce的概念
- 必须关于GPU方面的MapReduce实现
- Loading Balancing
- 算法必须是数值计算或图像处理


### Kmeans算法加速
#### 运⽤MapReduce和CUDA实现Kmeans算法的GPU版本。

#### Kmeans聚类算法流程：
1. ⾸先确定⼀个k值，即我们希望将数据集经过聚类得到k个集合。
2. 从数据集中随机选择k个数据点作为质⼼。
3. 对数据集中每⼀个点，计算其与每⼀个质⼼的距离（如欧式距离），离哪个质⼼近，就划分到那个质⼼所属的集
合。
4. 把所有数据归好集合后，⼀共有k个集合。然后重新计算每个集合的质⼼。
5. 如果新计算出来的质⼼和原来的质⼼之间的距离⼩于某⼀个设置的阈值（表⽰重新计算的质⼼的位置变化不⼤，趋
于稳定，或者说收敛），我们可以认为聚类已经达到期望的结果，算法终⽌。
6. 如果新质⼼和原质⼼距离变化很⼤，需要迭代3~5步骤。
算法相关参数
K值：要得到的簇的个数
质⼼：每个簇的均值向量，即向量各维取平均即可
距离量度：常⽤欧⼏⾥得距离和余弦相似度（先标准化）

#### MapReduce核⼼思想
---分⽽治之

适⽤于⼤量复杂的任务处理场景（⼤规模数据处理场景）。
MapReduce处理的数据类型是<key,value>键值对。
- Map负责“分”，即把复杂的任务分解为若⼲个“简单的任务”来并⾏处理。可以进⾏拆分的前提是这些⼩任务可以并⾏计
算，彼此间⼏乎没有依赖关系。 \
此种场景可以使⽤CUDA调⽤GPU线程来模拟。编程时将处理任务拆解为**单指令多数据模式，即 SIMD编程模型**。 
- Reduce负责“合”，即对map阶段的结果进⾏全局汇总。 
这两种阶段合起来正是MapReduce思想的体现。

并⾏计算前提是如何划分计算任务或者计算数据以便对划分的⼦任务或数据块同时进⾏计算。
kmeans算法迭代过程中，涉及⼤量的距离计算，属于计算密集型任务，且数据可以分为具有同样计算过程的数据块，
块之间数据不存在数据依赖关系，具有良好的可并⾏性。利⽤CUDA加速算法，可以极⼤提⾼算法性能。

**实现流程：**
- 导⼊数据，初始化质⼼。
- 设置算法迭代次数iterations。
- 执⾏Mapper操作
  - 用GPU线程模拟节点，负责计算当前点与k个质⼼的距离，距离度量采⽤欧式距离。
  - 根据最⼩的距离将current point划分到对应的簇。
  - ⽣成key-value结构pairs。key->cluster_id, value->point
  - 执⾏线程同步。
- Shuffle阶段
  - 对所有的pairs键值对，按key值进⾏排序。
  - 对上⼀步⽣成的有序pairs进⾏分区，将相同的key(cluster_id)数据划分到同⼀分区。
- 执⾏Reducer操作
  - 分配reducer节点对pair中间结果进行汇总（此处如果按照cluster_id进行任务划分，容易导致负载倾斜）。
  - 调用GPU线程并行统计每个簇对应的数据量。
  - 根据每个簇的数据量动态分配线程。
  - 利用并行规约求得簇内所有点x,y坐标和，除以length得到平均值作为新的质⼼。
  - 执⾏线程同步。
- 不停迭代，直⾄收敛。
- 将device端运算结果拷⻉⾄host端，保存结果。

算法使⽤数据集Clustering basic benchmark Birch-sets \
数据集地址 http://cs.joensuu.fi/sipu/datasets/ \
birch1.txt (100000, 2)


Environment：

Ubuntu 16.04, VS Code

Languages: Cuda, C++, Makefile

Compiler: g++ -std=c++11, nvcc

Cuda version: 10.1
