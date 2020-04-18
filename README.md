# awesome multi-view clustering
Collections for state-of-the-art (SOTA), novel multi-view clustering methods (papers, codes and datasets)

We are looking forward for other participants to share their papers and codes. If interested, please contanct <wangsiwei13@nudt.edu.cn>.

##  Table of Contents
- [Surveys](#jump1) 
- [Papers and Codes](#jump2)
    - [Graph Clustering](#jump21)
    - [Multiple Kenrel Clustering (MKC)](#jump22)
    - [Subspace Clustering](#jump23)
    - [Deep Multi-view Clustering](#jump24)
    - [Binary Multi-view Clustering](#jump25)
- [Benchmark Datasets](#jump3)
    - [Oringinal Datasets](#jump31)
    - [Kernelized Datasets](#jump32)

---

##  <span id="jump1">Important Survey Papers </span>
1. A survey on multi-view learning [Paper](https://arxiv.org/pdf/1304.5634)

1. A study of graph-based system for multi-view clustering [Paper](https://www.researchgate.net/profile/Hao_Wang250/publication/328573967_A_study_of_graph-based_system_for_multi-view_clustering/links/5cbff7e5299bf120977adaa6/A-study-of-graph-based-system-for-multi-view-clustering.pdf) [code](https://github.com/cswanghao/gbs)

1. Multi-view clustering: A survey [Paper](https://ieeexplore.ieee.org/iel7/8254253/8336843/08336846.pdf)

1. Multi-view learning overview: Recent progress and new challenges [Paper](https://www.researchgate.net/profile/Shiliang_Sun2/publication/314251895_Multi-view_Learning_Overview_Recent_Progress_and_New_Challenges/links/5def9d8f92851c836470978c/Multi-view-Learning-Overview-Recent-Progress-and-New-Challenges.pdf)

---

## <span id="jump2">Papers </span>
Papers are listed in the following methods:graph clustering, NMF-based clustering, co-regularized, subspace clustering and multi-kernel clustering

### <span id="jump21">Graph Clusteirng</span> 
1. AAAI15: Large-Scale Multi-View Spectral Clustering via Bipartite Graph [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9641/9937) [code](https://github.com/zzz123xyz/MVSC)

1. TKDE2018： One-step multi-view spectral clustering [Paper](https://ieeexplore.ieee.org/abstract/document/8478288/) [code]()

1. TKDE19: GMC: Graph-based Multi-view Clustering [Paper](https://ieeexplore.ieee.org/abstract/document/8662703) [code](https://github.com/cshaowang/gmc)

1. ICDM2019: Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering [Paper](https://www.researchgate.net/profile/Dong_Huang9/publication/335857675_Consistency_Meets_Inconsistency_A_Unified_Graph_Learning_Framework_for_Multi-view_Clustering/links/5d809ca7458515fca16e3776/Consistency-Meets-Inconsistency-A-Unified-Graph-Learning-Framework-for-Multi-view-Clustering.pdf) [code](https://github.com/youweiliang/ConsistentGraphLearning)


### <span id="jump22">Multiple Kenrel Clustering(MKC)</span> 
1. NIPS14: Localized Data Fusion for Kernel k-Means Clustering with Application to Cancer Biology [Paper](https://papers.nips.cc/paper/5236-localized-data-fusion-for-kernel-k-means-clustering-with-application-to-cancer-biology.pdf) [code](https://github.com/mehmetgonen/lmkkmeans)

1. IJCAI15： Robust Multiple Kernel K-means using L21-norm [Paper](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/download/11332/11224) [code](https://github.com/csliangdu/RMKKM)

1. AAAI16：Multiple Kernel k-Means Clustering with Matrix-Induced Regularization [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/12115/11819) [code](https://github.com/wangsiwei2010/Multiple-Kernel-k-Means-Clustering-with-Matrix-Induced-Regularization)

1. IJCAI19:  Multi-view Clustering with Late Fusion Alignment Maximization [paper](https://www.ijcai.org/proceedings/2019/0524.pdf) [code](https://github.com/wangsiwei2010/latefusionalignment)

### <span id="jump23">Subspace Clustering</span> 
1. CVPR2015 Diversity-induced Multi-view Subspace Clustering [Paper](https://www.zpascal.net/cvpr2015/Cao_Diversity-Induced_Multi-View_Subspace_2015_CVPR_paper.pdf) [code](http://cic.tju.edu.cn/faculty/zhangchangqing/code/DiMSC.rar)

1. CVPR2017 Latent Multi-view Subspace Clustering [Paper](http://cic.tju.edu.cn/faculty/zhangchangqing/pub/Zhang_Latent_Multi-View_Subspace_CVPR_2017_paper.pdf) [code](http://cic.tju.edu.cn/faculty/zhangchangqing/code/LMSC_CVPR2017_Zhang.rar)

1. AAAI2018 Consistent and Specific Multi-view Subspace Clustering [Paper](https://github.com/XIAOCHUN-CAS/Academic-Publications/blob/master/Conference/2018_AAAI_Luo.pdf) [code](https://github.com/XIAOCHUN-CAS/Consistent-and-Specific-Multi-View-Subspace-Clustering)

1. PR2018: Multi-view Low-rank Sparse Subspace Clustering [Paper](https://arxiv.org/abs/1708.08732) [code](https://github.com/wangsiwei2010/Multi-view-LRSSC)

1. TIP2019: Split Multiplicative Multi-view Subspace Clustering [Paper](https://www.researchgate.net/publication/333007034_Split_Multiplicative_Multi-view_Subspace_Clustering) [code](https://github.com/joshuaas/SM2SC)

1. IJCAI19: Flexible multi-view representation learning for subspace clustering [Paper](https://www.ijcai.org/Proceedings/2019/0404.pdf) [code](https://github.com/lslrh/FMR)

1. ICCV19: Reciprocal Multi-Layer Subspace Learning for Multi-View Clustering [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Reciprocal_Multi-Layer_Subspace_Learning_for_Multi-View_Clustering_ICCV_2019_paper.pdf) [code](https://github.com/lslrh/RMSL)


### <span id="jump24">Deep Multi-view Clustering</span> 
1. CVPR2019:  AE^2-Nets: Autoencoder in Autoencoder Networks [Paper](http://cic.tju.edu.cn/faculty/zhangchangqing/pub/AE2_Nets.pdf) [code](https://github.com/willow617/AE2-Nets)

1. TIP2019: Multi-view Deep Subspace Clustering Networks [Paper](https://arxiv.org/abs/1908.01978) [code](https://github.com/huybery/MvDSCN)
1. TKDE2020: Joint Deep Multi-View Learning for Image Clustering [Paper](https://ieeexplore.ieee.org/abstract/document/8999493/)

### <span id="jump25">Binary Multi-view Clustering</span> 
1. TPAMI2019: Binary Multi-View Clustering [Paper](http://cfm.uestc.edu.cn/~fshen/TPAMI-BMVC_Final.pdf) [code](https://github.com/DarrenZZhang/BMVC)


---

## <span id="jump3">Benchmark Datasets</span>
### <span id="jump31">Oringinal Datasets</span>
### <span id="jump32">Kernelized Datasets</span>