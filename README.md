# Storage for AI

## Data Preparation

[2019 CLUSTER] **Efficient User-Level Storage Disaggregation for Deep Learning**. [[PDF](https://par.nsf.gov/servlets/purl/10156300)]

[2020 FAST] **Quiver: An Informed Storage Cache for Deep Learning**. [[PDF](https://www.usenix.org/system/files/fast20-kumar.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast20_slides_kumar.pdf)]

[2020 ICPP] **DIESEL: A Dataset-Based Distributed Storage and Caching System for Large-Scale Deep Learning Training**. [[PDF](https://dl.acm.org/doi/abs/10.1145/3404397.3404472)] [[Slides](https://jnamaral.github.io/icpp20/slides/Wang_DIESEL.pdf)]

[2021 VLDB] **Analyzing and Mitigating Data Stalls in DNN Training**. [[PDF](http://www.vldb.org/pvldb/vol14/p771-mohan.pdf)]

[2021 VLDB] **tf.data: A Machine Learning Data Processing Framework**. [[PDF](http://www.vldb.org/pvldb/vol14/p2945-klimovic.pdf)]

[2021 ATC] **Refurbish Your Training Data: Reusing Partially Augmented Samples for Faster Deep Neural Network Training**. [[PDF](https://www.usenix.org/system/files/atc21-lee.pdf)] [[Slides](https://www.usenix.org/system/files/atc21_slides_lee.pdf)]

[2022 SIGMOD] **Where Is My Training Bottleneck? Hidden Trade-Offs in Deep Learning Preprocessing Pipelines**. [[PDF](https://dl.acm.org/doi/10.1145/3514221.3517848)] [[Recording](https://youtu.be/md5NWGsMHeo)]

[2022 ATC] **Cachew: Machine Learning Input Data Processing as a Service**. [[PDF](https://www.usenix.org/system/files/atc22-graur.pdf)]

[2022 TPDS] **DIESEL+: Accelerating Distributed Deep Learning Tasks on Image Datasets**. [[PDF](https://doi.org/10.1109/TPDS.2021.3104252)]

[2023 ATC] **Tectonic-Shift: A Composite Storage Fabric for Large-Scale ML Training**. [[PDF](https://www.usenix.org/system/files/atc23-zhao.pdf)] [[Slides](https://www.usenix.org/system/files/atc23_slides_zhao.pdf)]

## Model Training & Inference

### Checkpointing

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

### Other

[2023 PVLDB] **Efficient Fault Tolerance for Recommendation Model Training via Erasure Coding**. [[PDF](https://www.vldb.org/pvldb/vol16/p3137-kosaian.pdf)] [[Code](https://github.com/Thesys-lab/ECRec)]

## Benchmark

[2020 MLSys] **MLPerf Training Benchmark**. [[PDF](https://proceedings.mlsys.org/paper/2020/file/02522a2b2726fb0a03bb19f2d8d9524d-Paper.pdf)]
