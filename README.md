# awesome multi-view clustering
Collections for state-of-the-art (SOTA), novel multi-view clustering methods (papers, codes and datasets)

We are looking forward for other participants to share their papers and codes. If interested, please contanct <wangsiwei13@nudt.edu.cn>.

##  Table of Contents
- [Surveys](#jump1) 
- [Papers and Codes](#jump2)
    - [Graph Clustering](#jump21)
    - [Multiple Kenrel Clustering (MKC)](#jump22)
    - [Subspace Clustering](#jump23)
    - [NMF-based Clustering](#jump26)
    - [Deep Multi-view Clustering](#jump24)
    - [Binary Multi-view Clustering](#jump25)
    - [Ensemble Multi-view Clustering](#jump27)
- [Benchmark Datasets](#jump3)
    - [Oringinal Datasets](#jump31)
    - [Kernelized Datasets](#jump32)
- [Scholars](#jump4)

---

##  <span id="jump1">Important Survey Papers </span>
1. Xu, Chang, Dacheng Tao, and Chao Xu. "A survey on multi-view learning" (2013) Cite: 897 [[Paper]](https://arxiv.org/pdf/1304.5634)
2. Wang, Hao, et al. "A study of graph-based system for multi-view clustering" (2019) Cite: 102 [[Paper]](https://www.researchgate.net/profile/Hao_Wang250/publication/328573967_A_study_of_graph-based_system_for_multi-view_clustering/links/5cbff7e5299bf120977adaa6/A-study-of-graph-based-system-for-multi-view-clustering.pdf) [[code]](https://github.com/cswanghao/gbs)
3. Yang, Yan, and Hao Wang. "Multi-view clustering: A survey" (2018) Cite: 117 [[Paper]](https://ieeexplore.ieee.org/iel7/8254253/8336843/08336846.pdf)
4. Zhao, Jing, et al. "Multi-view learning overview: Recent progress and new challenges" (2017) Cite: 419 [[Paper]](https://www.researchgate.net/profile/Shiliang_Sun2/publication/314251895_Multi-view_Learning_Overview_Recent_Progress_and_New_Challenges/links/5def9d8f92851c836470978c/Multi-view-Learning-Overview-Recent-Progress-and-New-Challenges.pdf)

---

## <span id="jump2">Papers </span>
Papers are listed in the following methods:graph clustering, NMF-based clustering, co-regularized, subspace clustering and multi-kernel clustering

### <span id="jump21">Graph Clusteirng</span> 

- AAAI15: 
  1. Li, Yeqing, et al. "Large-Scale Multi-View Spectral Clustering via Bipartite Graph" [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9641/9937) [[code]](https://github.com/zzz123xyz/MVSC)

- IJCAI17: 
  1. Nie, Feiping, Jing Li, and Xuelong Li. "Self-Weighted Multiview Clustering with Multiple Graphs" [[Paper]](https://www.ijcai.org/Proceedings/2017/0357.pdf) [[code]](https://github.com/kylejingli/SwMC-IJCAI17)

- TKDE2018：
  1. Zhu, Xiaofeng, et al. "One-step multi-view spectral clustering" [[Paper]](https://ieeexplore.ieee.org/abstract/document/8478288/) [[code]](https://pan.baidu.com/s/1eFiB87O0LBkJS8ZRSybNfQ)

- TKDE19: 
  1. Wang, Hao, Yan Yang, and Bing Liu. "GMC: Graph-based Multi-view Clustering" [[Paper]](https://ieeexplore.ieee.org/abstract/document/8662703) [[code]](https://github.com/cshaowang/gmc)

- ICDM2019: 
  1. Liang, Youwei, Dong Huang, and Chang-Dong Wang. "Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering" [[Paper]](https://www.researchgate.net/profile/Dong_Huang9/publication/335857675_Consistency_Meets_Inconsistency_A_Unified_Graph_Learning_Framework_for_Multi-view_Clustering/links/5d809ca7458515fca16e3776/Consistency-Meets-Inconsistency-A-Unified-Graph-Learning-Framework-for-Multi-view-Clustering.pdf) [[code]](https://github.com/youweiliang/ConsistentGraphLearning)


### <span id="jump22">Multiple Kenrel Clustering(MKC)</span> 
- NIPS14: 
  1. Gönen, Mehmet, and Adam A. Margolin. "Localized Data Fusion for Kernel k-Means Clustering with Application to Cancer Biology" [[Paper]](https://papers.nips.cc/paper/5236-localized-data-fusion-for-kernel-k-means-clustering-with-application-to-cancer-biology.pdf) [[code]](https://github.com/mehmetgonen/lmkkmeans)

- IJCAI15： 
  1. Du, Liang, et al. "Robust Multiple Kernel K-means using L21-norm" [[Paper]](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/download/11332/11224) [[code]](https://github.com/csliangdu/RMKKM)

- AAAI16：
  1. Liu, Xinwang, et al. "Multiple Kernel k-Means Clustering with Matrix-Induced Regularization" [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/12115/11819) [[code]](https://github.com/wangsiwei2010/Multiple-Kernel-k-Means-Clustering-with-Matrix-Induced-Regularization)

- IJCAI19:  
  1. Wang, Siwei, et al. "Multi-view Clustering with Late Fusion Alignment Maximization" [[Paper]](https://www.ijcai.org/proceedings/2019/0524.pdf) [[code]](https://github.com/wangsiwei2010/latefusionalignment)

- TNNLS2019: 
  1. Zhou, Sihang, et al. "Multiple kernel clustering with neighbor-kernel subspace segmentation" [[Paper]](https://ieeexplore.ieee.org/document/8750871) [[code]](https://github.com/SihangZhou/Demo-of-Multiple-Kernel-Clustering-with-Neighbor-Kernel-Subspace-Segmentation)

### <span id="jump23">Subspace Clustering</span> 
- CVPR2015: 
  1. Cao, Xiaochun, et al. "Diversity-induced Multi-view Subspace Clustering" [[Paper]](https://www.zpascal.net/cvpr2015/Cao_Diversity-Induced_Multi-View_Subspace_2015_CVPR_paper.pdf) [[code]](http://cic.tju.edu.cn/faculty/zhangchangqing/code/DiMSC.rar)

- CVPR2017:
  1. Zhang, Changqing, et al. "Latent Multi-view Subspace Clustering" [[Paper]](http://cic.tju.edu.cn/faculty/zhangchangqing/pub/Zhang_Latent_Multi-View_Subspace_CVPR_2017_paper.pdf) [[code]](http://cic.tju.edu.cn/faculty/zhangchangqing/code/LMSC_CVPR2017_Zhang.rar)

- AAAI2018:
  1. Luo, Shirui, et al. "Consistent and Specific Multi-view Subspace Clustering" [[Paper]](https://github.com/XIAOCHUN-CAS/Academic-Publications/blob/master/Conference/2018_AAAI_Luo.pdf) [[code]](https://github.com/XIAOCHUN-CAS/Consistent-and-Specific-Multi-View-Subspace-Clustering)

- PR2018: 
  1. Brbić, Maria, and Ivica Kopriva. "Multi-view Low-rank Sparse Subspace Clustering" [[Paper]](https://arxiv.org/abs/1708.08732) [[code]](https://github.com/wangsiwei2010/Multi-view-LRSSC)

- TIP2019: 
  1. Yang, Zhiyong, et al. "Split Multiplicative Multi-view Subspace Clustering" [[Paper]](https://www.researchgate.net/publication/333007034_Split_Multiplicative_Multi-view_Subspace_Clustering) [[code]](https://github.com/joshuaas/SM2SC)

- IJCAI19: 
  1. Li, Ruihuang, et al. "Flexible multi-view representation learning for subspace clustering" [[Paper]](https://www.ijcai.org/Proceedings/2019/0404.pdf) [[code]](https://github.com/lslrh/FMR)

- ICCV19:
  1. Li, Ruihuang, et al. "Reciprocal Multi-Layer Subspace Learning for Multi-View Clustering" [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Reciprocal_Multi-Layer_Subspace_Learning_for_Multi-View_Clustering_ICCV_2019_paper.pdf) [[code]](https://github.com/lslrh/RMSL)


### <span id="jump24">Deep Multi-view Clustering</span> 
- CVPR2019:  
  1. Zhang, Changqing, Yeqing Liu, and Huazhu Fu. "AE^2-Nets: Autoencoder in Autoencoder Networks" [[Paper]](http://cic.tju.edu.cn/faculty/zhangchangqing/pub/AE2_Nets.pdf) [[code]](https://github.com/willow617/AE2-Nets)

- TIP2019:
  1. Zhu, Pengfei, et al. "Multi-view Deep Subspace Clustering Networks" [[Paper]](https://arxiv.org/abs/1908.01978) [[code]](https://github.com/huybery/MvDSCN)

- ICML2019: 
  1. Peng, Xi, et al. "COMIC: Multi-view Clustering Without Parameter Selection" [[paper]](http://proceedings.mlr.press/v97/peng19a/peng19a.pdf) [[code]](https://github.com/limit-scu/2019-ICML-COMIC)

- IJCAI2019: 
  1. Huang, Zhenyu, et al. "Multi-view Spectral Clustering Network" [[paper]](https://www.ijcai.org/Proceedings/2019/0356.pdf) [[code]](https://github.com/limit-scu/2019-IJCAI-MvSCN)
  2. Li, Zhaoyang, et al. "Deep Adversarial Multi-view Clustering Network" [[paper]](https://www.ijcai.org/Proceedings/2019/0409.pdf) [[code]](https://github.com/IMKBLE/DAMC)

- TKDE2020: 
  1. Xie, Yuan, et al. "Joint Deep Multi-View Learning for Image Clustering" [[Paper]](https://ieeexplore.ieee.org/abstract/document/8999493/)

### <span id="jump25">Binary Multi-view Clustering</span> 
- TPAMI2019: 
  1. Zhang, Zheng, et al. "Binary Multi-View Clustering" [[Paper]](http://cfm.uestc.edu.cn/~fshen/TPAMI-BMVC_Final.pdf) [[code]](https://github.com/DarrenZZhang/BMVC)


### <span id="jump26">NMF-based Multi-view Clustering</span> 
- AAAI20: 
  1. Chen, Man-Sheng, et al. "Multi-view Clustering in Latent Embedding Space" [[Paper]](https://www.researchgate.net/profile/Dong_Huang9/publication/338883065_Multi-view_Clustering_in_Latent_Embedding_Space/links/5e30e4ee458515072d6ab048/Multi-view-Clustering-in-Latent-Embedding-Space.pdf?_sg%5B0%5D=c7_LGDqrWNZ_2R_YVqZW5paGs4aiAWHyL5Vm6D9xC-qLrwZgnT5PnHd5qcLIWLjUU1w1sMRvcFieskwMXfiUxA.C7MpmX3wox2zTGV_rHjWvJVYUcWBn5cx271Yud84FlPQiu_W8azOItQWDVbvUiM3bw4kxI_zLS8mGKTKMl5f3w&_sg%5B1%5D=Ug4z3sxpjLL5fvIFDmpbr9hht6CQIYTxXEPWuPHRJZvOOuGvEI2QyxzM8WX0M3c0SkQeyoVq3fnE9kyqH5TWHTslmLrQDWSN3t6xvMVZkLTi.C7MpmX3wox2zTGV_rHjWvJVYUcWBn5cx271Yud84FlPQiu_W8azOItQWDVbvUiM3bw4kxI_zLS8mGKTKMl5f3w&_iepl=) [[code]](https://github.com/Ttuo123/MCLES)


### <span id="jump27"> Ensemble-based Multi-view Clustering</span> 
- TNNLS2019: 
  1. Tao, Zhiqiang, et al. "Marginalized Multiview Ensemble Clustering" [[Paper]](https://ieeexplore.ieee.org/document/8691702) [[code]](https://pan.baidu.com/s/1ipfGlQKcBTQn71yP3ZbISQ)



---

## <span id="jump3">Benchmark Datasets</span>
### <span id="jump31">Oringinal Datasets</span>
1. It contains seven widely-used multi-view datasets: Handwritten (HW), Caltech-7/20, BBCsports, Nuswide, ORL and Webkb. Released by Baidu Service.
[address](https://pan.baidu.com/s/1hG2zL40RxVaJ_p53gBM7kA) （code）gaih


| Name of dataset | Samples | Views | Clusters | Original   location                                                                                                                                           |                                             |   |   |
|-----------------|---------|-------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|---|---|
| Handwritten     | 2000    | 6     | 10       |                                                                                                                                                               |                                             |   |   |
| Caltech-7       | 1474    | 6     | 7        | http://www.vision.caltech.edu/Image_Datasets/Caltech101/                                                                                                      |                                             |   |   |
| Caltech-20      | 2386    | 6     | 20       | http://www.vision.caltech.edu/Image_Datasets/Caltech101/                                                                                                      |                                             |   |   |
| BBCsports       | 3183    | 2     | 5        | http://mlg.ucd.ie/datasets/segment.html                                                                                                                       |                                             |   |   |
| Nuswide         | 30000   | 5     | 31       | https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html                                                                            |                                             |   |   |
| ORL             | 400     | 3     | 40       | http://www.uk.research.att.com/facedatabase.html                                                                                                              |                                             |   |   |
| Webkb           | 1051    | 2     | 2        | http://www.cs.cmu.edu/afs/cs/project/theo-11/www/wwkb/                                                                                                        | http://membres-lig.imag.fr/grimal/data.html |   |   |
| Cornell         | 165     | 2     | 15       | http://membres-lig.imag.fr/grimal/data.html                                                                                                                   |                                             |   |   |
| MSRC-v1         | 210     | 6     | 7        | https://www.microsoft.com/en-us/research/project/image-understanding/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fobjectclassrecognition%2F |                                             |   |   |
| Wikipedia       | 693     | 2     | 10       | http://www.svcl.ucsd.edu/projects/crossmodal/                                                                                                                 |                                             |   |   |
| BBCsport        | 116     | 4     | 5        | http://mlg.ucd.ie/datasets/segment.html                                                                                                                       | http://mlg.ucd.ie/datasets/bbc.html         |   |   |
| yaleA           | 165     | 3     | 15       | http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html                                                                                                     |                                             |   |   |
| mfeat           | 2000    | 6     | 10       | http://archive.ics.uci.edu/ml/datasets/Multiple+Features                                                                                                      |                                             |   |   |
| aloi            | 110250  | 8     | 1000     | http://elki.dbs.ifi.lmu.de/wiki/DataSets/MultiView                                                                                                            |                                             |   |   |

### <span id="jump32">Kernelized Datasets</span>
1. The following kernelized datasets are created by our team. For more information, you can ask <wangsiwei13@nudt.edu.cn> for help.
[address](https://pan.baidu.com/s/1sOpNOG_3BlNPoxhwLKbUEQ) （code）y44e 

If you use our code or datasets, please cite our with the following bibtex code :
```
@inproceedings{wang2019multi,
  title={Multi-view clustering via late fusion alignment maximization},
  author={Wang, Siwei and Liu, Xinwang and Zhu, En and Tang, Chang and Liu, Jiyuan and Hu, Jingtao and Xia, Jingyuan and Yin, Jianping},
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence},
  pages={3778--3784},
  year={2019},
  organization={AAAI Press}
}
```

---

## <span id="jump2">Scholars </span>

- [Feiping Nie](https://sites.google.com/site/feipingnie/publications), from Tsinghua University, cited by 22874
- [Xinwang Liu](https://xinwangliu.github.io/), from National University of Defense Technology (NUDT), cited by 4133.
- [Changqing Zhang](http://cic.tju.edu.cn/faculty/zhangchangqing/research.html), from Tianjin University, cited by 3096.
- [Xi Peng](http://pengxi.me/), from Sichuan Univeristy (SCU), cited by 2460.
