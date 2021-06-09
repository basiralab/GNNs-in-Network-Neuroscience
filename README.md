# Graph Neural Network for Network Neuroscience

> In Geometric Deep Learning (GDL), one of the most popular learning methods is the Graph Neural Network (GNN), which applies convolutional layers to learn the topological structure of the input graph. GNN has recently been used for the analysis of different types of the human connectome, such as structural, functional, and morphological networks derived respectively from Diffusion Tensor Imaging (DTI), functional magnetic resonance imaging (fMRI), and T1-w MRI data. Such a nascent field has achieved significant performance improvements over traditional deep neural networks in the early diagnosis of brain disorders. Thus, we list here all MICCAI papers of 4 years (2017-2020), some journals, and IPMI (2017-2019) papers that are embedded into multiple tasks related to the disease diagnosis:

* [Predicting one or multiple modalities from a a single or different modalities (e.g., data synthesis)](#SingleMultiviewprediction)
* [Predicting high-resolution data from low-resolution data](#resolution)
* [Predicting the brain evolution trajectory](#timedependent)
* [Multiple network inegration and fusion](#Integrationfusion)
* [Computer-aided prognostic methods (e.g., for brain diseases)](#diseaseclassification)
* [Biomarker Identification](#biomarker)
* [Extra works related to machine learning](#ML)

If you like to update the file by adding the unlisted open-source articles, feel free to open an issue or submit a pull requests. You can also directly contact Alaa Bessadok at alaa.bessadok@gmail.com. All contributions are very welcome! 

<a name="SingleMultiviewprediction" />

# arXiv link

The full paper is downloadable at 
https://arxiv.org/pdf/2106.03535.pdf

## Cross-domain graph prediction

### Single graph prediction
| Title                                                        | Paper | Author | Dataset | Code | Youtube Video |  Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Recovering Brain Structural Connectivity from Functional Connectivity via Multi-GCN Based Generative Adversarial Network | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_6)  | Lu Zhang | [HCP](https://www.humanconnectome.org/study/hcp-young-adult/data-releases)   | __ | __ | MICCAI 2020 | 
| Topology-Guided Cyclic Brain Connectivity Generation using Geometric Deep Learning| [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165027020304118) | Abubakhari Sserwadda | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | [Python](https://github.com/basiralab/CGTS-GAN) | __ | Journal of Neuroscience Methods 2020
| Deep Representation Learning for Multimodal Brain Networks | [ARXIV](https://arxiv.org/abs/2007.09777)  | Wen Zhang | [WU-Minn HCP](https://pubmed.ncbi.nlm.nih.gov/23684880/)   | __ | __ | MICCAI 2020  
| Symmetric Dual Adversarial Connectomic Domain Alignment for Predicting Isomorphic Brain Graph from a Baseline Graph | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_51)  | Alaa Bessadok |  [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | MICCAI 2019 
| Hierarchical Adversarial Connectomic Domain Alignment for Target Brain Graph Prediction and Classification from a Source Graph | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32281-6_11)  | Alaa Bessadok |  [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | [Python](https://github.com/basiralab/HADA) | [14min](https://www.youtube.com/watch?v=OJOtLy9Xd34) | PRIME-MICCAI 2019 
| Brain graph synthesis by dual adversarial domain alignment and target graph prediction from a source graph | [LNCS](https://www.sciencedirect.com/science/article/pii/S1361841520302668?via%3Dihub)  | Alaa Bessadok |  [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | Medical Image Analysis Journal 2021 

### Multi-graph prediction
| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Topology-Aware Generative Adversarial Network for Joint Prediction of Multiple Brain Graphs from a Single Brain Graph | [ARXIV](https://arxiv.org/abs/2009.11058)  | Alaa Bessadok | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | [Python]( https://github.com/basiralab/MultiGraphGAN) | [10min](https://www.youtube.com/watch?v=vEnzMQqbdHc) | MICCAI 2020

<a name="resolution" />

## Cross-resolution graph prediction

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| GSR-Net: Graph Super-Resolution Network for Predicting High-Resolution from Low-Resolution Functional Brain Connectomes | [ARXIV](https://arxiv.org/abs/2009.11080)  | Megi Isallari  |  [Southwest University Longitudinal Imaging Multimodal (SLIM) study](https://www.nature.com/articles/sdata201717)   | [Python](https://github.com/basiralab/GSR-Net) | [11min](https://www.youtube.com/watch?v=xwHKRxgMaEM) | MLMI-MICCAI 2020

<a name="timedependent" />

## Cross-time graph prediction

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Deep EvoGraphNet Architecture for Time-Dependent Brain Graph Data Synthesis from a Single Timepoint | [ARXIV](https://arxiv.org/abs/2009.13217)  | Ahmed Nebli  |  [OASIS-2](https://www.oasis-brains.org/)   | [Python](https://github.com/basiralab/EvoGraphNet) | [6min](https://www.youtube.com/watch?v=aT---t2OBO0) | PRIME-MICCAI 2020 
| Residual Embedding Similarity-Based Network Selection for Predicting Brain Network Evolution Trajectory from a Single Observation  | [ARXIV](https://arxiv.org/abs/2009.11110)  | Ahmet Serkan Goktas | [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/) | [Python](https://github.com/basiralab/RESNets) | [6min](https://www.youtube.com/watch?v=UOUHe-1FfeY) | PRIME-MICCAI 2020 
| Foreseeing Brain Graph Evolution over Time Using Deep Adversarial Network Normalizer | [ARXIV](https://arxiv.org/abs/2009.11166)  | Zeynep Gurler  | [OASIS-2](https://www.oasis-brains.org/)   | [Python](https://github.com/basiralab/gGAN) | [6min](https://www.youtube.com/watch?v=5vpQIFzf2Go) | PRIME-MICCAI 2020 



<a name="Integrationfusion" />

## Integration/Fusion

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------:  
| Deep Graph Normalizer: A Geometric Deep Learning Approach for Estimating Connectional Brain Templates | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_16)  | Mustafa Burak Gurbuz | [ADNI](http://adni.loni.usc.edu/) [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) | [Python]( https://github.com/basiralab/DGN) | [10min](https://www.youtube.com/watch?v=Q_WLY2ZNxRk) | MICCAI 2020  
| Clustering-Based Deep Brain MultiGraph Integrator Network for Learning Connectional Brain Templates | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-60365-6_11)  | Uğur Demir | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | [Python]( https://github.com/basiralab/cMGINet) | __ | GRAIL-MICCAI 2020 
<!--| Integrating Heterogeneous Brain Networks for Predicting Brain Disease Conditions | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_24)  | Yanfu Zhang  | [PPMİ](http://www.ppmi-info.org)  | __ | __ | MICCAI 2019 -->
<!--| Deep Representation Learning for Multimodal Brain Networks | [ARXIV](https://arxiv.org/abs/2007.09777)  | Wen Zhang | [WU-Minn HCP](https://pubmed.ncbi.nlm.nih.gov/23684880/)   | __ | __ | MICCAI 2020 -->
<!--| A Cascaded Multi-modality Analysis in Mild Cognitive Impairment | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_64) | Lu Zhang | [ADNI 3](http://adni.loni.usc.edu/)  | __ | __ | MLMI-MICCAI 2019-->
<!--| Unified Brain Network with Functional and Structural Data | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_12)  | Biopoint Autism Spectral Disorder dataset  | [Python](https://github.com/xxlya/PRGNN_fMRI) | __ | MICCAI 2020 -->
<!--| Persistent Feature Analysis of Multimodal Brain Networks Using Generalized Fused Lasso for EMCI Identification | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_5)  | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | MICCAI 2020 -->





<!--| Multi-view Brain HyperConnectome AutoEncoder for Brain State Classification | [LNCS](https://link.springer.com/10.1007/978-3-030-59728-3_8)  | Alin Banka | [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/)   | [Python](https://github.com/basiralab/HCAE) | [5min](https://www.youtube.com/watch?v=ncPyj_4cSe8) | PRIME-MICCAI 2020 
| Interpretable Multimodality Embedding of Cerebral Cortex Using Attention Graph Network for Identifying Bipolar Disorder | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_89) | Huzheng Yang | __  | __ | __ | MICCAI 2019
| Temporal-Adaptive Graph Convolutional Network for Automated Identification of Major Depressive Disorder Using Resting-State fMRI | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_1) | Dongren Yao | [MDD](https://www.pnas.org/content/116/18/9078)  | __ | __ | MLMI 2020-->

<!--| Brain Network Analysis and Classification Based on Convolutional Neural Network | [Journal website](https://www.frontiersin.org/articles/10.3389/fncom.2018.00095/full) | Lu Meng | [MEG dataset](https://www.sciencedirect.com/science/article/pii/S221315821500100X?via%3Dihub) | __ | __ | Frontiers in Computational Neuroscience  2018-->


<!-- | Connectome Prior in Deep Neural Networks to Predict Autism  | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8363534)  | Colin J. Brown | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) | __ | __ | ISBI 2018-->
<!-- BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment -->
<!--| Transport-Based Joint Distribution Alignment for Multi-site Autism Spectrum Disorder Diagnosis Using Resting-State fMRI | [LNCS](https://link.springer.com/chapter/10.1007%2F978-3-030-59713-9_43)  | Junyi Zhang  | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | MICCAI 2020 -->
<!--| Adaptive Functional Connectivity Network Using Parallel Hierarchical BiLSTM for MCI Diagnosis | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_58) | Yiqiao Jiang | [ADNI](http://adni.loni.usc.edu/)  | __ | __ | MLMI-MICCAI 2019-->



<a name="diseaseclassification" />

## Brain graph classification

### Graph embedding-based methods


| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------:  
| Multi-view Brain HyperConnectome AutoEncoder for Brain State Classification | [LNCS](https://arxiv.org/abs/2009.11553)  | Alin Banka  | [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/)   | [Python](https://github.com/basiralab/HCAE) | [6min](https://www.youtube.com/watch?v=ncPyj_4cSe8) | PRIME-MICCAI 2020 
| Adversarial Connectome Embedding for Mild Cognitive Impairment Identification Using Cortical Morphological Networks | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32391-2_8)  | Alin Banka |  [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/)   | __ | __ | CIN-MICCAI 2019 
| Deep Hypergraph U-Net for Brain Graph Embedding and Classification | [ARXIV](https://arxiv.org/abs/2008.13118)  | Mert Lostar |  [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/)   | [Python](https://github.com/basiralab/HUNet) | __ | 2020
| Interpretable Multimodality Embedding of Cerebral Cortex Using Attention Graph Network for Identifying Bipolar Disorder | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_89) | Huzheng Yang | __  | __ | __ | MICCAI 2019
| New Graph-Blind Convolutional Network for Brain Connectome Data Analysis  | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_52)  | Yanfu Zhang | [HCP](https://www.humanconnectome.org)  | __ | __ | IPMI 2019
| Attention-Diffusion-Bilinear Neural Network for Brain Network Analysis | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32070948/)  | Jiashuang Huang | Epilepsy  dataset | __ | __ | IEEE Transactions on Medical Imaging 2020


### Graph-based methods

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Metric learning with spectral graph convolutions on brain connectivity networks  | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811917310765)  | Sofia Ira Ktena | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)  [UK Biobank] | [Python](https://github.com/sk1712/gcn_metric_learning) | __ | Neuroimage Journal 2018
| Dynamic Spectral Graph Convolution Networks with Assistant Task Training for Early MCI Diagnosis | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_70) | Xiaodan Xing | [ADNI2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2544629/)   | __ | __ | MICCAI 2019
| A Cascaded Multi-modality Analysis in Mild Cognitive Impairment | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_64) | Lu Zhang | [ADNI 3](http://adni.loni.usc.edu/)  | __ | __ | MLMI-MICCAI 2019
| DS-GCNs: Connectome Classification Using Dynamic Spectral Graph Convolution Networks with Assistant Task Training  | [ARXIV](https://arxiv.org/pdf/2001.03057.pdf)  | Xiaodan Xing | [ABIDE]( http://adni.loni.usc.edu)  | __ | __ | Cerebral Cortex 2021
| Multi-Hops Functional Connectivity Improves Individual Prediction of Fusiform Face Activation via a Graph Neural Network| [Journal website](https://www.frontiersin.org/articles/10.3389/fnins.2020.596109/full) | Dongya Wu | [HCP](https://www.humanconnectome.org) | __ | __ | Frontiers in Neuroscience 2021

<!--| Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis  | [Journal website](https://www.frontiersin.org/articles/10.3389/fnins.2020.00630/full)  | Byung-Hoon Kim  | [HCP](https://www.humanconnectome.org)   | __ | __ | Frontiers in Neuroscience 2020-->
<!--| A deep spatiotemporal graph learning architecture for brain connectivity analysis  | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9175360)  | Tiago Azevedo | [HCP](https://www.humanconnectome.org)   | __ | __ | 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC 2020)-->

### Population-based methods

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Integrating Similarity Awareness and Adaptive Calibration in Graph Convolution Network to Predict Disease | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_13)  | Xuegang Song  | [ADNI](http://adni.loni.usc.edu/) | __ | __ | MICCAI 2020 
| Spectral Graph Convolutions for Population-Based Disease Prediction   | [ARXIV](https://arxiv.org/abs/1703.03020)  | Sarah Parisot  | [ADNI](http://adni.loni.usc.edu/) [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)  | [Python](https://github.com/parisots/population-gcn) | __ | MICCAI 2017
| InceptionGCN: Receptive Field Aware Graph Convolutional Network for Disease Prediction  | [ARXIV](https://arxiv.org/abs/1903.04233)  | Anees Kazi | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) [TADPOLE](http://adni.loni.usc.edu/)  | __ | __ | IPMI 2019



<a name="#biomarker" />

## Biomarker identification

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Graph Neural Network for Interpreting Task-fMRI Biomarkers | [ARXIV](https://arxiv.org/abs/1907.01661)  | Xiaoxiao Li |  [UNC/UMN Baby Connectome Project](https://pubmed.ncbi.nlm.nih.gov/29578031/)   | [Yale Child Study Center] | __ | MICCAI 2019
| Pooling Regularized Graph Neural Network for fMRI Biomarker Analysis | [ARXIV](https://arxiv.org/abs/2007.14589)  | Xiaoxiao Li  | multi-modal epilepsy dataset   | __ | __ | MICCAI 2020
| Interpretable Multimodality Embedding of Cerebral Cortex Using Attention Graph Network for Identifying Bipolar Disorder | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_89) | Huzheng Yang | __  | __ | __ | MICCAI 2019
| BrainGNN: Interpretable Brain Graph NeuralNetwork for fMRI Analysis| [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.05.16.100057v1) | Xiaoxiao Li | [Biopoint](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5510046/) [HCP](https://www.humanconnectome.org) | __ | __ | bioRxiv 2020

<!--| Generalizable Machine Learning inNeuroscience using Graph Neural Networks | [ARXIV](https://arxiv.org/abs/2010.08569)  | Paul Y. Wang  | [Kato dataset](https://www.sciencedirect.com/science/article/pii/S0092867415011964?via%3Dihub)   | __ | __ | Frontiers in Neuroscience 2021 -->
<!--| A Multi-Domain Connectome Convolutional Neural Network for Identifying Schizophrenia From EEG Connectivity Patterns  | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8836535)  | Chun-Ren Phang | [HCP](https://www.humanconnectome.org/study/hcp-young-adult/data-releases)  | __ | __ | IEEE Journal of Biomedical and Health Informatics 2020-->
<!--| Disentangled Intensive Triplet Autoencoder for Infant Functional Connectome Fingerprinting | [LNCS](https://link.springer.com/10.1007/978-3-030-59728-3_8)  | Lu Meng  | [UNC/UMN Baby Connectome Project](https://pubmed.ncbi.nlm.nih.gov/29578031/)   | __ | __ | MICCAI 2020 -->
<!--| Distance Metric Learning Using Graph Convolutional Networks: Application to Functional Brain Networks  | [ARXIV](https://arxiv.org/abs/1703.02161)  | Sofia Ira Ktena | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)    | [Python](https://github.com/sk1712/gcn_metric_learning) | __ | MICCAI 2017-->







<a name="ML" />

## Machine Learning and other types of algorithm for Network Neuroscience

## Single/Multi-view prediction

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Multi-view Brain Network Prediction from a Source View Using Sample Selection via CCA-Based Multi-kernel Connectomic Manifold Learning  | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-00320-3_12)  | Minghui Zhu | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | PRIME-MICCAI 2018

## Integration/Fusion

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Supervised Multi-topology Network Cross-Diffusion for Population-Driven Brain Network Atlas Estimation | [ARXIV](https://arxiv.org/abs/2009.11054)  | Islem Mhiri | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | [Python]( https://github.com/basiralab/SM-netFusion-PY) [MATLAB]( https://github.com/basiralab/SM-netFusion) | [10min](https://www.youtube.com/watch?v=eWz65SyR-eM) | MICCAI 2020 
| Multi-scale Profiling of Brain Multigraphs by Eigen-Based Cross-diffusion and Heat Tracing for Brain State Profiling | [ARXIV](https://arxiv.org/abs/2009.11534)  | Mustafa Saglam | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ |  [8min](https://www.youtube.com/watch?v=D_E2m6O37mk) | GRAIL-MICCAI 2020 
| Estimation of Brain Network Atlases UsingDiffusive-Shrinking Graphs:Application to Developing Brains | [LNCS](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_31) | Islem Rekik |  UNC/UMN Baby Connectome dataset   | __ |  __ | IPMI 2017 

## Resolution

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Predicting High-Resolution Brain Networks Using Hierarchically Embedded and Aligned Multi-resolution Neighborhoods | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32281-6_12)  | Kübra Cengiz | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)  | __ | __ | PRIME-MICCAI 2019

## Time-dependent

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Progressive Infant Brain Connectivity Evolution Prediction from Neonatal MRI Using Bidirectionally Supervised Sample Selection | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32281-6_7)  | Olfa Ghribi | __ | __ | __ | MICCAI 2019
| Learning-Guided Infinite Network Atlas Selection for Predicting Longitudinal Brain Network Evolution from a Single Observation | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_88)  | Baha Eddine Ezzine  | [ADNI GO](https://pubmed.ncbi.nlm.nih.gov/16443497/)   | __ | __ | MICCAI 2019

## Disease classification

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Graph Morphology-Based Genetic Algorithm for Classifying Late Dementia States | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-32391-2_3) | Oumaima Ben Khelifa  | __  | __ | __ | CNI-MICCAI 2019
| Joint Correlational and Discriminative Ensemble Classifier Learning for Dementia Stratification Using Shallow Brain Multiplexes | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-00928-1_68)  | Rory Raeper  | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)   | __ | __ | PRIME-MICCAI 2018

## Other predictions

| Title                                                        | Paper | Author | Dataset | Code | Youtube Video | Proceeding/Journal/Year |
|:------------------------------------------------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:| :----------------------:|:----------------------: 
| Predicting Emotional Intelligence Scores from Multi-session Functional Brain Connectomes | [LNCS](https://link.springer.com/chapter/10.1007/978-3-030-00320-3_13)  | Anna Lisowska | [SLIM](https://www.nature.com/articles/sdata201717) | __ | __ | PRIME-MICCAI 2018

<!-- ## [Intresting but not related to the current task Survey/Review papers]
A Survey on Deep Learning for Neuroimaging-Based Brain Disorder Analysis = https://www.frontiersin.org/articles/10.3389/fnins.2020.00779/full
Deep learning for neuroimaging: a validation study =  https://www.frontiersin.org/articles/10.3389/fnins.2014.00229/full !-->

<!-- ## [Great for inspiration and for the writing] 
Geometric Convolutional Neural Network for Analyzing Surface-Based Neuroimaging Data = https://www.frontiersin.org/articles/10.3389/fninf.2018.00042/full -->

