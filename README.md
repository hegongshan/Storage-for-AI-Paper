# Storage for AI

* [1.Data Preparation](#data-preparation)
    * [1.1 Storage Format](#storage-format)
    * [1.2 Storage System](#storage-system)
    * [1.3 Caching System](#caching-system)
    * [1.4 Data Pipeline](#data-pipeline)
* [2.Model Training & Inference](#model-training-&-inference)
    * [2.1 Fault Tolerance](#fault-tolerance)
        * [2.1.1 Checkpointing](#checkpointing)
        * [2.1.2 Others](#others)
    * [2.2 Model Storage System](#model-storage-system)
    * [2.3 KV Cache](#kv-cache)
* [3.Benchmark](#benchmark)
* [4.Profiling/Analysis Tool](#profilinganalysis-tool)
* [5.Survey](#survey)

## Data Preparation

### Storage Format

[2019 Big Data] WebDataset: **High Performance I/O For Large Scale Deep Learning**. [[PDF](https://arxiv.org/pdf/2001.01858)] [[Poster](https://storagetarget.com/wp-content/uploads/2019/12/deep-learning-large-scale-phys-poster-1.pdf)] [[Code](https://github.com/webdataset/webdataset)]

[2021 PVLDB] **Progressive Compressed Records: Taking a Byte out of Deep Learning Data**. [[PDF](https://vldb.org/pvldb/vol14/p2627-kuchnik.pdf)] [[Code](https://github.com/mkuchnik/PCR_Release)]

[2022 ECCV] **L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training**. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710171.pdf)]

[2023 CVPR] **FFCV: Accelerating Training by Removing Data Bottlenecks**. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Leclerc_FFCV_Accelerating_Training_by_Removing_Data_Bottlenecks_CVPR_2023_paper.pdf)] [[Code](https://github.com/libffcv/ffcv)]

[2023 CIDR] **Deep Lake: a Lakehouse for Deep Learning**. [[PDF](https://www.cidrdb.org/cidr2023/papers/p69-buniatyan.pdf)]

[2024 PACMMOD] **The Image Calculator: 10x Faster Image-AI Inference by Replacing JPEG with Self-designing Storage Format**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3639307)]

[2025 CIDR] **Frequency-Store: Scaling Image AI by A Column-Store for Images**. [[PDF](https://www.vldb.org/cidrdb/papers/2025/p27-sirin.pdf)]

### Storage System

[2019 CLUSTER] DLFS: **Efficient User-Level Storage Disaggregation for Deep Learning**. [[PDF](https://par.nsf.gov/servlets/purl/10156300)]

[2019 Big Data] AIStore: **High Performance I/O For Large Scale Deep Learning**. [[PDF](https://arxiv.org/pdf/2001.01858)] [[Poster](https://storagetarget.com/wp-content/uploads/2019/12/deep-learning-large-scale-phys-poster-1.pdf)] [[Code](https://github.com/NVIDIA/aistore)]

[2020 ICPP] **DIESEL: A Dataset-Based Distributed Storage and Caching System for Large-Scale Deep Learning Training**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3404397.3404472)] [[Slides](https://jnamaral.github.io/icpp20/slides/Wang_DIESEL.pdf)]

[2022 TPDS] **DIESEL+: Accelerating Distributed Deep Learning Tasks on Image Datasets**. [[PDF](https://doi.org/10.1109/TPDS.2021.3104252)]

[2023 ATC] **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**. [[PDF](https://www.usenix.org/system/files/atc23-zhao.pdf)] [[Slides](https://www.usenix.org/system/files/atc23_slides_zhao.pdf)]

[2024 SC] 3FS: **Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning**. [[PDF](https://arxiv.org/pdf/2408.14158)] [[Code](https://github.com/deepseek-ai/3FS)]

[2025 ASPLOS] **OS2G: A High-Performance DPU Offloading Architecture for GPU-based Deep Learning with Object Storage**. [[PDF](https://doi.org/10.1145/3676641.3716265)]

### Caching System

[2020 FAST] **Quiver: An Informed Storage Cache for Deep Learning**. [[PDF](https://www.usenix.org/system/files/fast20-kumar.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast20_slides_kumar.pdf)]

[2021 SC] NoPFS: **Clairvoyant Prefetching for Distributed Machine Learning I/O**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3458817.3476181)] [[Slides](http://vwgwjkk.unixer.de/publications/img/dryden-nopfs-slides.pdf)] [[Code](https://github.com/spcl/NoPFS)]

[2022 CLUSTER] **Hvac: Removing I/O Bottleneck for Large-Scale Deep Learning Applications**. [[PDF](https://doi.org/10.1109/CLUSTER51413.2022.00044)]

[2023 FAST] **SHADE: Enable Fundamental Cacheability for Distributed Deep Learning Training**. [[PDF](https://www.usenix.org/system/files/fast23-khan.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast23_slides_khan.pdf)] [[Code](https://github.com/rkhan055/SHADE)]

[2023 HPCA] **iCache: An Importance-Sampling-Informed Cache for Accelerating I/O-Bound DNN Model Training**. [[PDF](http://www.cs.iit.edu/~scs/assets/files/chen2023icache.pdf)] [[Slides](http://cs.iit.edu/~scs/assets/files/chen2023icache-slides.pdf)] [[Code](https://github.com/ISCS-ZJU/iCache)]

[2025 DATE] **LCache: Log-Structured SSD Caching for Training Deep Learning Models**. [[PDF](https://doi.org/10.23919/DATE64628.2025.10992907)]

### Data Pipeline

[2021 PVLDB] **Analyzing and Mitigating Data Stalls in DNN Training**. [[PDF](http://www.vldb.org/pvldb/vol14/p771-mohan.pdf)] [[DS-Analyzer](https://github.com/msr-fiddle/DS-Analyzer)] [[CoorDL Code](https://github.com/msr-fiddle/CoorDL)]

[2021 PVLDB] **tf.data: A Machine Learning Data Processing Framework**. [[PDF](http://www.vldb.org/pvldb/vol14/p2945-klimovic.pdf)]

[2021 ATC] **Refurbish Your Training Data: Reusing Partially Augmented Samples for Faster Deep Neural Network Training**. [[PDF](https://www.usenix.org/system/files/atc21-lee.pdf)] [[Slides](https://www.usenix.org/system/files/atc21_slides_lee.pdf)]

[2022 SIGMOD] **Where Is My Training Bottleneck? Hidden Trade-Offs in Deep Learning Preprocessing Pipelines**. [[PDF](https://dl.acm.org/doi/10.1145/3514221.3517848)] [[Recording](https://youtu.be/md5NWGsMHeo)] [[Code](https://github.com/cirquit/presto)]

[2022 ATC] **Cachew: Machine Learning Input Data Processing as a Service**. [[PDF](https://www.usenix.org/system/files/atc22-graur.pdf)] [[Code](https://github.com/eth-easl/cachew)]

[2022 NeurIPS] Joader: **A Deep Learning Dataloader with Shared Data Preparation**. [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/6d538a6e667960b168d3d947eb6207a6-Paper-Conference.pdf)]

[2023 CVPR] **FFCV: Accelerating Training by Removing Data Bottlenecks**. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Leclerc_FFCV_Accelerating_Training_by_Removing_Data_Bottlenecks_CVPR_2023_paper.pdf)] [[Code](https://github.com/libffcv/ffcv)]

[2023 SoCC] **tf.data service: A Case for Disaggregating ML Input Data Processing**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3620678.3624666)]

[2023 PACMMOD] **GoldMiner: Elastic Scaling of Training Data Pre-Processing Pipelines for Deep Learning**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3589773)]

[2023 PVLDB] **FastFlow: Accelerating Deep Learning Model Training with Smart Offloading of Input Data Pipeline**. [[PDF](https://www.vldb.org/pvldb/vol16/p1086-um.pdf)]

[2024 TC] **MMDataLoader: Reusing Preprocessed Data Among Concurrent Model Training Tasks**. [[PDF](https://doi.org/10.1109/TC.2023.3336161)]

[2024 ATC] **Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement**. [[PDF](https://www.usenix.org/system/files/atc24-graur.pdf)] [[Code](https://github.com/eth-easl/cachew/tree/pecan)]

[2025 PVLDB] **cedar: Optimized and Unified Machine Learning Input Data Pipelines**. [[PDF](https://www.vldb.org/pvldb/vol18/p488-zhao.pdf)] [[Code](https://github.com/stanford-mast/cedar)]

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

[2025 ASPLOS] **MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training**. [[PDF](https://dl.acm.org/doi/10.1145/3676641.3716006)] [[Slides](https://jyhuang91.github.io/talks/asplos2025-moc-system-slides.pdf)]

[2025 EuroSys] **FlowCheck: Decoupling Checkpointing and Training of Large-Scale Models**. [[PDF](https://dl.acm.org/doi/10.1145/3689031.3696088)] [[Code](https://github.com/AlibabaResearch/flowcheck-eurosys25)]

[2025 PVLDB] **IncrCP: Decomposing and Orchestrating Incremental Checkpoints for Effective Recommendation Model Training**. [[PDF](https://www.vldb.org/pvldb/vol18/p1049-du.pdf)] [[Code](https://github.com/linqy71/IncrCP_paper)]

[2025 NSDI] **ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development**. [[PDF](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)] [[Code](https://github.com/ByteDance-Seed/ByteCheckpoint)]

[2025 ATC] **Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism**. [[Code](https://github.com/deepspeedai/DeepSpeed)]

#### Others

[2023 NSDI] **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs**. [[PDF](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)] [[Slides](https://www.usenix.org/system/files/nsdi23_slides_thorpe-john.pdf)] [[Code](https://github.com/uclasystem/bamboo)]

[2023 SOSP] **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3600006.3613152)] [[Code](https://github.com/SymbioticLab/Oobleck)]

[2023 PVLDB] **Efficient Fault Tolerance for Recommendation Model Training via Erasure Coding**. [[PDF](https://www.vldb.org/pvldb/vol16/p3137-kosaian.pdf)] [[Code](https://github.com/Thesys-lab/ECRec)]

[2024 TPDS] **Swift: Expedited Failure Recovery for Large-scale DNN Training**. [[PDF](https://i2.cs.hku.hk/~cwu/papers/yczhong-tpds24.pdf)] [[Code](https://github.com/jasperzhong/swift)]

[2024 SOSP] **ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3694715.3695960) [[Slides](https://swapnilgandhi.com/slides/recycle-sosp24.pdf)] [[Poster](https://swapnilgandhi.com/posters/ReCycle_SOSP.pdf)]

### Model Storage System

[2023 ICS] **DStore: A Lightweight Scalable Learning Model Repository with Fine-Grain Tensor-Level Access**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3577193.3593730)]

[2024 HPDC] **EvoStore: Towards Scalable Storage of Evolving Learning Models**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3625549.3658679)]

[2025 ICDE] **MLKV: Efficiently Scaling up Large Embedding Model Training with Disk-Based Key-Value Storage**. [[PDF](https://www.computer.org/csdl/proceedings-article/icde/2025/360300e134/26FZCltmgdq)] [[Code](https://github.com/llm-db/MLKV)]

### KV Cache

[2023 SOSP] vLLM: **Efficient Memory Management for Large Language Model Serving with PagedAttention**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)] [[Code](https://github.com/vllm-project/vllm)]

[2024 OSDI] **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**. [[PDF](https://www.usenix.org/system/files/osdi24-lee.pdf)] [[Slides](https://www.usenix.org/system/files/osdi24_slides-lee.pdf)] [[Code](https://github.com/snu-comparch/InfiniGen)]

[2024 ATC] **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**. [[PDF](https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf)]

[2024 ACL] **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**. [[PDF](https://aclanthology.org/2024.acl-long.623.pdf)] [[Code](https://github.com/microsoft/chunk-attention)]

[2025 FAST] **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**. [[PDF](https://www.usenix.org/system/files/fast25-qin.pdf)] [[Slides](https://www.usenix.org/system/files/fast25_slides-qin.pdf)] [[Code](https://github.com/kvcache-ai/Mooncake)]

[2025 FAST] **IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**. [[PDF](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)] [[Slides](https://www.usenix.org/system/files/fast25_slides-chen-weijian-impress.pdf)]

[2025 EuroSys] **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**. [[PDF](https://doi.org/10.1145/3689031.3696098)] [[Code](https://github.com/LMCache/LMCache)]

[2025 ASPLOS] **vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3669940.3707256)] [[Code](https://github.com/microsoft/vattention)]

[2025 ASPLOS] FlashGen: **Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3676641.3716245)] [[Code](https://jinujeong.github.io/slides/ASPLOS25_FlashGen_Slide.pdf)]

## Benchmark

[2021 CCGrid] **DLIO: A Data-Centric Benchmark for Scientific Deep Learning Applications**. [[PDF](http://www.cs.iit.edu/~scs/assets/files/devarajan2021dlio.pdf)] [[Code](https://github.com/argonne-lcf/dlio_benchmark)]

## Profiling/Analysis Tool

[2020 CLUSTER] **tf-Darshan: Understanding Fine-grained I/O Performance in Machine Learning Workloads**. [[PDF](https://arxiv.org/pdf/2008.04395)]

## Survey

[2024 JCRD] **From BERT to ChatGPT: Challenges and Technical Development of Storage Systems for Large Model Training** (in Chinese). [[PDF](https://dx.doi.org/10.7544/issn1000-1239.202330554)]

[2025 ACM Computing Surveys] **I/O in Machine Learning Applications on HPC Systems: A 360-degree Survey**. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3722215)]

[2025 JCRD] **Survey of Storage Optimization Techniques in Large Language Model Inference** (in Chinese). [[PDF](https://dx.doi.org/10.7544/issn1000-1239.202440628)]
