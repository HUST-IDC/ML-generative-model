
## Sampling方法介绍

### 背景知识：
现实中，很多问题无法用分析的方法来求得精确解，需要通过一些方法得到近似解，随机模拟就是一种**求近似解**的方法。

假设，我们有一个矩形的区域R（大小已知），在这个区域中有一个不规则的区域M（即不能通过公式直接计算出来），现在要求取M的面积？ 怎么求？近似的方法很多，例如：把这个不规则的区域M划分为很多很多个小的规则区域，用这些规则区域的面积求和来近似M，另外一个近似的方法就是采样的方法，我们抓一把黄豆，把它们均匀地铺在矩形区域，如果我们知道黄豆的总个数S，那么只要我们数数位于不规则区域M中的黄豆个数S1，那么我们就可以求出M的面积：M=S1*R/S

随机模拟方法的核心就是**如何对一个概率分布得到样本**，即采样（sampling）。而Sampling方法解决问题的基本思想，就是**产生一系列样本来模拟一个概率分布**。


###  1. 蒙特卡洛数值积分

如果我们要求f(x)的积分，而f(x)的形式可能比较复杂，积分不好求，则可以通过数值解法来求近似的结果。常用的方法是蒙特卡洛积分：

$$\int_a^b f(x) dx = \int_a^b \frac{f(x)}{g(x)}g(x) dx = \frac{1}{n}\sum _1^n \frac{f(x)}{g(x)}$$


这样把g(x)看做是x在区间内的概率分布，而把前面的分数部分看做一个函数，然后在g(x)下**抽取n个样本**，当n足够大时，可以用采用均值来近似。因此只要g(x)比较容易采到数据样本，就可以求得分f(x)在区间上的积分。

### 2. Monte Carlo principle

Monte Carlo 采样计算：x表示随机变量，服从概率分布 p(x)，那么要计算 f(x) 的期望，只需要我们不停从 p(x) 中抽样xi，然后对这些f(xi)取平均即可近似f(x)的期望。

![enter image description here](https://images0.cnblogs.com/blog/533521/201310/25225400-30083dce288f4bbfbd0294d8c70e553b.png)

![enter image description here](https://images0.cnblogs.com/blog/533521/201310/25225413-7405b98e045b4af09eea448fb1db4eb5.gif)

### 3. 接受-拒绝抽样（Acceptance-Rejection sampling)

很多实际问题中，p(x)是很难直接采样的的，因此，我们需要求助其他的手段来采样。既然 p(x) 太复杂在程序中没法直接采样，那么我设定一个程序可抽样的分布 q(x) 比如高斯分布，然后按照一定的方法拒绝某些样本，达到接近 p(x) 分布的目的:

![enter image description here](https://images0.cnblogs.com/blog/533521/201310/25225434-fd6db018b45d4152a09ea1de2b5304ad.png)

具体操作如下，设定一个方便抽样的函数 q(x)，以及一个常量 k，使得 p(x) 总在 kq(x) 的下方。（参考上图）

-   x 轴方向：从 q(x) 分布抽样得到 a。
-   y 轴方向：从均匀分布(0, kq(a)) 中抽样得到 u。
-   如果刚好落到灰色区域: u > p(a) 拒绝， 否则接受这次抽样
-   重复以上过程

在高维的情况下，接受-拒绝采样会出现两个问题：第一是合适的 q 分布比较难以找到，第二是很难确定一个合理的 k 值。这两个问题会导致拒绝率很高，无用计算增加。

### 4. 重要性抽样(Importance sampling)

与接受-拒绝采样一样，重要性采样同样借助了一个易于采样的q(x)：

![enter image description here](https://images0.cnblogs.com/blog/533521/201310/25225454-745161c7386a4a88bb04fe3d52691994.png)

其中，$$ \frac{p(z)}{q(z)} $$ 可以看做 importance weight。我们来考察一下上面的式子，p 和 f 是确定的，我们要确定的是 q。要确定一个什么样的分布才会让采样的效果比较好呢？直观的感觉是，样本的方差越小期望收敛速率越快。比如一次采样是 0, 一次采样是 1000, 平均值是 500,这样采样效果很差，如果一次采样是 499, 一次采样是 501, 你说期望是 500,可信度还比较高。在上式中，我们目标是p×f/q方差越小越好，所以 |p×f| 大的地方，proposal distribution q(z) 也应该大。

但是可惜的是，在高维空间里找到一个这样合适的 q 非常难。因为根据上面的小方差最优原理，我们经常会取一些简单的分布作为q。但是当x是高维数据的时候，q分布的简单性很难与p或者pf相匹配。当q>>pf时候，重要采样采到了很多无用的样本（权值之和很小，或趋近于0）。当q<<pf时候，样本很少被采集到，其对应的权值会非常大。

### 5. 马尔科夫链  Markov Chain

马尔科夫链的数学定义:           
![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132334569283.png)


也就是说前一个状态只与当前状态有关，而与其他状态无关，Markov Chain 体现的是状态空间的转换关系，下一个状态只决定与当前的状态。

举例来说，社会学家经常把人按其经济状况分成3类：下层(lower-class)、中层(middle-class)、上层(upper-class)，我们用1,2,3 分别代表这三个阶层。社会学家们发现决定一个人的收入阶层的最重要的因素就是其父母的收入阶层。如果一个人的收入属于下层类别，那么他的孩子属于下层收入的概率是 0.65, 属于中层收入的概率是 0.28, 属于上层收入的概率是 0.07。事实上，从父代到子代，收入阶层的变化的转移概率如下：

![enter image description here](https://uploads.cosx.org/2013/01/table-1.jpg)
![enter image description here](https://uploads.cosx.org/2013/01/markov-transition.png)

转换成转移矩阵的形式：
![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132350037556.png)


![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132380665858.png)


![enter image description here](https://uploads.cosx.org/2013/01/table-2.jpg)


我们发现从第7代人开始，这个分布就稳定不变了，事实上，在这个问题中，从任意初始概率分布开始都会收敛到这个上面这个稳定的结果。

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132433319734.png)


对于给定的概率分布p(x),我们希望能有便捷的方式生成它对应的样本。由于马氏链能收敛到平稳分布， 于是一个很的漂亮想法是：如果我们能构造一个转移矩阵为P的马氏链，使得该马氏链的平稳分布恰好是p(x), 那么我们从任何一个初始状态x0出发沿着马氏链转移, 得到一个转移序列 x0,x1,x2,⋯xn,xn+1⋯,， 如果马氏链在第n步已经收敛了，于是我们就得到了 π(x) 的样本xn,xn+1⋯

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132453786092.png)


![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132487534424.png)


![enter image description here](https://uploads.cosx.org/2013/01/mcmc-transition.jpg)

假设我们已经有一个转移矩阵Q(对应元素为q(i,j)), 把以上的过程整理一下，我们就得到了如下的用于采样概率分布p(x)的算法。

![enter image description here](https://uploads.cosx.org/2013/01/mcmc-algo-1.jpg)

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012132534877269.png)


![enter image description here](https://uploads.cosx.org/2013/01/mcmc-algo-2.jpg)


### 6. 吉布斯采样 Gibbs Sampling

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012133000505875.png)


![enter image description here](https://uploads.cosx.org/2013/01/gibbs-transition.png)

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012133008472518.png)


![enter image description here](https://uploads.cosx.org/2013/01/gibbs-algo-1.jpg)

![enter image description here](https://uploads.cosx.org/2013/01/two-stage-gibbs.png)

![enter image description here](https://images0.cnblogs.com/blog/354318/201502/012133041447591.png)


![enter image description here](https://uploads.cosx.org/2013/01/gibbs-algo-2.jpg)

以上算法收敛后，得到的就是概率分布p(x1,x2,⋯,xn)的样本，当然这些样本并不独立，但是我们此处要求的是采样得到的样本符合给定的概率分布，并不要求独立。同样的，在以上算法中，坐标轴轮换采样不是必须的，可以在坐标轴轮换中引入随机性，这时候转移矩阵 Q 中任何两个点的转移概率中就会包含坐标轴选择的概率，而在通常的 Gibbs Sampling 算法中，坐标轴轮换是一个确定性的过程，也就是在给定时刻t，在一根固定的坐标轴上转移的概率是1。



**refrences：**

https://www.cnblogs.com/xbinworld/p/4266146.html

https://blog.csdn.net/xianlingmao/article/details/7768833

http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling


