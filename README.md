<div align="left">

# Toward Semantically Enhanced Representation Learning for Text-Based Person Retrieval (TIP 2026)

</div>

<p align="left">
Official PyTorch Implementation of Toward Semantically Enhanced Representation Learning for Text-Based Person Retrieval
</p>

<p align="left">
  <a href="https://ieeexplore.ieee.org/abstract/document/11493654">📄 Paper</a>
</p>

---

# Abstract

Text-Based Person Retrieval (TBPR) is a pivotal task in intelligent surveillance systems, aiming to retrieve target pedestrian images using free-form natural language descriptions. Existing methods attempt to enhance cross-modal alignment through multi-granularity interactions; however, their performance is still constrained by two critical challenges: cross-modal semantic inconsistency and insufficient semantic discriminability. To address these limitations, we propose **DSEE (Diversity Semantic Embedding Expansion)**, a novel framework for semantically enhanced representation learning. Unlike previous approaches that rely on constructing larger-scale or heavily annotated datasets, DSEE establishes identity-centric cross-modal consistency through contrastive learning and generative semantic synergy. Specifically, DSEE consists of two key modules: **Bidirectional-guided Semantic Modeling (BSM)** and **Generative-driven Semantic Enhancement (GSE)**. The BSM module constructs enriched semantic embeddings by modeling similarity-aware interactions between visual and textual modalities. By emphasizing identity-level semantic consistency, the module enhances both semantic expressiveness and discriminative alignment across modalities. Furthermore, the GSE module introduces semantic diversity via a vision-guided generative text augmentation strategy, while a dual-path attention mechanism jointly performs intramodal refinement and cross-modal semantic alignment to improve semantic precision and robustness. Extensive experiments on multiple public benchmarks demonstrate that DSEE achieves state-of-the-art performance under both standard and challenging evaluation settings. Our work provides an effective paradigm for robust and semantically discriminative TBPR representation learning in real-world surveillance applications.

---

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/figure1.png" width="55%">
</p>


# Framework Overview

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/figure2.png" width="95%">
</p>

---

# Datasets

Please download the datasets from the following official repositories:

| Dataset | Link |
|---|---|
| CUHK-PEDES | <a href="https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description">Download</a> |
| ICFG-PEDES | <a href="https://github.com/zifyloo/SSAN">Download</a> |
| RSTPReid | <a href="https://github.com/NjtechCVLab/RSTPReid-Dataset">Download</a> |

---

# Experimental Results

## CUHK-PEDES

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/cuhk.png" width="90%">
</p>


---

## ICFG-PEDES 

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/icfg.png" width="50%">
</p>

---

## RSTPReid

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/rstp.png" width="50%">
</p>


---

# Visualization Results

## Cross-modal Attention Visualization

<p align="center">
  <img src="https://github.com/jie4438/DSEE/blob/main/figure/figure8.png" width="95%">
</p>

---

# Citation

If you find this project useful for your research, please consider citing:

```bibtex
@ARTICLE{DSEE,
  author={Chen, Kun and Yang, Xi and Wang, Nannan},
  journal={IEEE Transactions on Image Processing}, 
  title={Toward Semantically Enhanced Representation Learning for Text-Based Person Retrieval}, 
  year={2026},
  volume={35},
  pages={4161-4176},
  doi={10.1109/TIP.2026.3684811}}
```

---

<div align="center">

</div>

</div>
