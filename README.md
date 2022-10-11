### Storage for AI

#### Fetch & Preprocessing

[2022 ATC] **Cachew: Machine Learning Input Data Processing as a Service**. [[PDF](https://www.usenix.org/system/files/atc22-graur.pdf)]

[2021 ATC] **Refurbish Your Training Data: Reusing Partially Augmented Samples for Faster Deep Neural Network Training**. [[PDF](https://www.usenix.org/system/files/atc21-lee.pdf)] [[Slides](https://www.usenix.org/system/files/atc21_slides_lee.pdf)]

> Preprocessing Stall: cache partially augmented samples across all epochs within a job

[2021 VLDB] **Analyzing and Mitigating Data Stalls in DNN Training**. [[PDF](http://www.vldb.org/pvldb/vol14/p771-mohan.pdf)]

> Hyperparameter (HP) Search: stage preprocessed minibatch across all HP jobs *within an epoch*

[2020 FAST] **Quiver: An Informed Storage Cache for Deep Learning**. [[PDF](https://www.usenix.org/system/files/fast20-kumar.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast20_slides_kumar.pdf)]

> Fetch Stall (Remote): share cached training data among multiple tasks

#### Checkpointing

[2022 NSDI] **Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models**. [[PDF](https://www.usenix.org/system/files/nsdi22-paper-eisenman.pdf)] [[Slides](https://www.usenix.org/system/files/nsdi22_slides_eisenman.pdf)]

[2021 FAST] **CheckFreq: Frequent, Fine-Grained DNN Checkpointing**. [[PDF](https://www.usenix.org/system/files/fast21-mohan.pdf)] [[Slides](https://www.usenix.org/sites/default/files/conference/protected-files/fast21_slides_mohan.pdf)]

[2020 CCGRID] **DeepFreeze: Towards Scalable Asynchronous Checkpointing of Deep Learning Models**. [[PDF](https://web.cels.anl.gov/~woz/papers/DeepFreeze_2020.pdf)]

#### Data Pipeline

[2021 VLDB] **tf.data: A Machine Learning Data Processing Framework**. [[PDF](http://www.vldb.org/pvldb/vol14/p2945-klimovic.pdf)]

#### Other

[2021 Ph.D. Dissertation] **Accelerating Deep Learning Training : A Storage Perspective**. [[PDF](https://repositories.lib.utexas.edu/bitstream/handle/2152/115131/MOHAN-DISSERTATION-2021.pdf)]

#### Benchmark

[2020 MLSys] **MLPerf Training Benchmark**. [[PDF](https://proceedings.mlsys.org/paper/2020/file/02522a2b2726fb0a03bb19f2d8d9524d-Paper.pdf)]

[2021 Big Data Mining And Analytics] **AIPerf: Automated Machine Learning as an AI-HPC Benchmark**. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9430136)]