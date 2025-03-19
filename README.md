# Storage for AI

* [1.Data Preparation](#data-preparation)
* [2.Model Training & Inference](#model-training-&-inference)
    * [2.1 Fault Tolerance](#fault-tolerance)
        * [2.1.1 Checkpointing](#checkpointing)
        * [2.1.2 Others](#others)
    * [2.2 Model Repository](#model-repository)
    * [2.3 KV Cache](#kv-cache)
* [3.Benchmark](#benchmark)
* [4.Profiling/Analysis Tool](#profilinganalysis-tool)
* [5.Survey](#survey)

## Data Preparation

[2019 CLUSTER] **Efficient User-Level Storage Disaggregation for Deep Learning**. [[PDF](https://par.nsf.gov/servlets/purl/10156300)]

[2020 FAST] **Quiver: An Informed Storage Cache for Deep Learning**. [[PDF](https://www.usenix.org/system/files/fast20-kumar.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast20_slides_kumar.pdf)]

[2020 ICPP] **DIESEL: A Dataset-Based Distributed Storage and Caching System for Large-Scale Deep Learning Training**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3404397.3404472)] [[Slides](https://jnamaral.github.io/icpp20/slides/Wang_DIESEL.pdf)]

[2021 VLDB] **Analyzing and Mitigating Data Stalls in DNN Training**. [[PDF](http://www.vldb.org/pvldb/vol14/p771-mohan.pdf)] [[DS-Analyzer](https://github.com/msr-fiddle/DS-Analyzer)] [[CoorDL Code](https://github.com/msr-fiddle/CoorDL)]

[2021 VLDB] **tf.data: A Machine Learning Data Processing Framework**. [[PDF](http://www.vldb.org/pvldb/vol14/p2945-klimovic.pdf)]

[2021 ATC] **Refurbish Your Training Data: Reusing Partially Augmented Samples for Faster Deep Neural Network Training**. [[PDF](https://www.usenix.org/system/files/atc21-lee.pdf)] [[Slides](https://www.usenix.org/system/files/atc21_slides_lee.pdf)]

[2022 SIGMOD] **Where Is My Training Bottleneck? Hidden Trade-Offs in Deep Learning Preprocessing Pipelines**. [[PDF](https://dl.acm.org/doi/10.1145/3514221.3517848)] [[Recording](https://youtu.be/md5NWGsMHeo)] [[Code](https://github.com/cirquit/presto)]

[2022 ATC] **Cachew: Machine Learning Input Data Processing as a Service**. [[PDF](https://www.usenix.org/system/files/atc22-graur.pdf)] [[Code](https://github.com/eth-easl/cachew)]

[2022 TPDS] **DIESEL+: Accelerating Distributed Deep Learning Tasks on Image Datasets**. [[PDF](https://doi.org/10.1109/TPDS.2021.3104252)]

[2022 CLUSTER] **Hvac: Removing I/O Bottleneck for Large-Scale Deep Learning Applications**. [[PDF](https://doi.org/10.1109/CLUSTER51413.2022.00044)]

[2023 FAST] **SHADE: Enable Fundamental Cacheability for Distributed Deep Learning Training**. [[PDF](https://www.usenix.org/system/files/fast23-khan.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast23_slides_khan.pdf)] [[Code](https://github.com/rkhan055/SHADE)]

[2023 HPCA] **iCache: An Importance-Sampling-Informed Cache for Accelerating I/O-Bound DNN Model Training**. [[PDF](http://www.cs.iit.edu/~scs/assets/files/chen2023icache.pdf)] [[Slides](http://cs.iit.edu/~scs/assets/files/chen2023icache-slides.pdf)] [[Code](https://github.com/ISCS-ZJU/iCache)]

[2023 ATC] **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**. [[PDF](https://www.usenix.org/system/files/atc23-zhao.pdf)] [[Slides](https://www.usenix.org/system/files/atc23_slides_zhao.pdf)]

[2023 SoCC] **tf.data service: A Case for Disaggregating ML Input Data Processing**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3620678.3624666)]

[2024 ATC] **Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement**. [[PDF](https://www.usenix.org/system/files/atc24-graur.pdf)] [[Code](https://github.com/eth-easl/cachew/tree/pecan)]

## Model Training & Inference

### Fault Tolerance

#### Checkpointing

[2020 CCGRID] **DeepFreeze: Towards Scalable Asynchronous Checkpointing of Deep Learning Models**. [[PDF](https://web.cels.anl.gov/~woz/papers/DeepFreeze_2020.pdf)]

[2020 ICML] **On Efficient Constructions of Checkpoints**. [[PDF](https://proceedings.mlr.press/v119/chen20m/chen20m.pdf)]

[2020 ICPP] **Delta-DNN: Efficiently Compressing Deep Neural Networks via Exploiting Floats Similarity**. [[PDF](https://par.nsf.gov/servlets/purl/10158379)]

[2021 FAST] **CheckFreq: Frequent, Fine-Grained DNN Checkpointing**. [[PDF](https://www.usenix.org/system/files/fast21-mohan.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast21_slides_mohan.pdf)] [[Code](https://github.com/msr-fiddle/CheckFreq)]

[2021 ICCD] **QD-Compressor: a Quantization-based Delta Compression Framework for Deep Neural Networks**. [[PDF](https://www.researchgate.net/profile/Xiangyu-Zou-4/publication/356082777_QD-Compressor_a_Quantization-based_Delta_Compression_Framework_for_Deep_Neural_Networks/links/618b29c73068c54fa5c8664a/QD-Compressor-a-Quantization-based-Delta-Compression-Framework-for-Deep-Neural-Networks.pdf)]

[2022 NSDI] **Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models**. [[PDF](https://www.usenix.org/system/files/nsdi22-paper-eisenman.pdf)] [[Slides](https://www.usenix.org/system/files/nsdi22_slides_eisenman.pdf)]

[2023 TPDS] **Design of a Quantization-Based DNN Delta Compression Framework for Model Snapshots and Federated Learning**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10018182)]

[2023 SOSP] **GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints**. [[PDF](https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf)] [[Slides](https://zhuangwang93.github.io/docs/Gemini_SOSP23_slides.pdf)]

[2023 ICCD] **A Cost-Efficient Failure-Tolerant Scheme for Distributed DNN Training**. [[PDF](https://csyhua.github.io/csyhua/hua-iccd2023-LightCheck.pdf)] [[Slides](https://csyhua.github.io/csyhua/hua-iccd2023-LightCheck-slide.pdf)] [[Code](https://github.com/LighT-chenml/LightCheck)]

[2024 EuroSys] **Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3627703.3650085)]

[2024 ICDCS] **Portus: Efficient DNN Checkpointing to Persistent Memory with Zero-Copy**. [[PDF](https://www.tianyuanwu.com/files/portus.pdf)]

[2024 HPDC] **DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3625549.3658685)] [[Code](https://github.com/DataStates/datastates-llm)]

[2024 ICML] **ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking**. [[PDF](https://openreview.net/pdf?id=hlvKd7Vdxm)] [[Code](https://github.com/Gaffey/ExCP)]

[2024 ICCD] **ParaCkpt: Heterogeneous Multi-Path Checkpointing Mechanism for Training Deep Learning Models**. [[PDF](https://doi.org/10.1109/ICCD63220.2024.00036)]

[2024 SoCC] **Inshrinkerator: Compressing Deep Learning Training Checkpoints via Dynamic Quantization**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3698038.3698553)]

[2024 TMC] **CheckBullet: A Lightweight Checkpointing System for Robust Model Training on Mobile Networks**. [[PDF](https://doi.org/10.1109/TMC.2024.3450283)]

[2025 FGCS] COCI: **Convergence-aware optimal checkpointing for exploratory deep learning training jobs**. [[PDF](https://doi.org/10.1016/j.future.2024.107597)] [[Code](https://github.com/wangzc-HPC/COCI)]

[2025 FCS] **BAFT: bubble-aware fault-tolerant framework for distributed DNN training with hybrid parallelism**. [[PDF](https://journal.hep.com.cn/fcs/EN/article/downloadArticleFile.do?attachType=PDF&id=35982)]

[2025 ASPLOS] **PCcheck: Persistent Concurrent Checkpointing for ML**. [[PDF](https://fotstrt.github.io/files/2025-pccheck.pdf)] [[Code](https://github.com/eth-easl/pccheck)]

#### Others

[2023 NSDI] **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs**. [[PDF](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)] [[Slides](https://www.usenix.org/system/files/nsdi23_slides_thorpe-john.pdf)] [[Code](https://github.com/uclasystem/bamboo)]

[2023 SOSP] **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3600006.3613152)] [[Code](https://github.com/SymbioticLab/Oobleck)]

[2023 PVLDB] **Efficient Fault Tolerance for Recommendation Model Training via Erasure Coding**. [[PDF](https://www.vldb.org/pvldb/vol16/p3137-kosaian.pdf)] [[Code](https://github.com/Thesys-lab/ECRec)]

[2024 TPDS] **Swift: Expedited Failure Recovery for Large-scale DNN Training**. [[PDF](https://i2.cs.hku.hk/~cwu/papers/yczhong-tpds24.pdf)] [[Code](https://github.com/jasperzhong/swift)]

[2024 SOSP] **ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3694715.3695960) [[Slides](https://swapnilgandhi.com/slides/recycle-sosp24.pdf)] [[Poster](https://swapnilgandhi.com/posters/ReCycle_SOSP.pdf)]

### Model Repository

[2023 ICS] **DStore: A Lightweight Scalable Learning Model Repository with Fine-Grain Tensor-Level Access**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3577193.3593730)]

[2024 HPDC] **EvoStore: Towards Scalable Storage of Evolving Learning Models**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3625549.3658679)]

### KV Cache

[2023 SOSP] **Efficient Memory Management for Large Language Model Serving with PagedAttention**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)] [[Code](https://github.com/vllm-project/vllm)]

[2025 FAST] **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**. [[PDF](https://www.usenix.org/system/files/fast25-qin.pdf)] [[Slides](https://www.usenix.org/system/files/fast25_slides-qin.pdf)] [[Code](https://github.com/kvcache-ai/Mooncake)]

[2025 FAST] **IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**. [[PDF](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)] [[Slides](https://www.usenix.org/system/files/fast25_slides-chen_weijian_impress.pdf)]

## Benchmark

[2021 CCGrid] **DLIO: A Data-Centric Benchmark for Scientific Deep Learning Applications**. [[PDF](http://www.cs.iit.edu/~scs/assets/files/devarajan2021dlio.pdf)] [[Code](https://github.com/argonne-lcf/dlio_benchmark)]

## Profiling/Analysis Tool

[2020 CLUSTER] **tf-Darshan: Understanding Fine-grained I/O Performance in Machine Learning Workloads**. [[PDF](https://arxiv.org/pdf/2008.04395)]

## Survey

[2024 JCRD] **From BERT to ChatGPT: Challenges and Technical Development of Storage Systems for Large Model Training** (in Chinese). [[PDF](https://dx.doi.org/10.7544/issn1000-1239.202330554)]

[2025 ACM Computing Surveys] **I/O in Machine Learning Applications on HPC Systems: A 360-degree Survey**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3722215)]

[2025 JCRD] **Survey of Storage Optimization Techniques in Large Language Model Inference** (in Chinese). [[PDF](https://dx.doi.org/10.7544/issn1000-1239.202440628)]
