<h1 align="center"><strong>HYDRA</strong></h1>
<h3 align="center"><em>A Hybrid Heuristic-Guided Deep Representation Architecture for Predicting Latent Zero-Day Vulnerabilities in Patched Functions</em></h3>

**HYDRA** is a hybrid vulnerability analysis framework designed to uncover latent **zero-day vulnerabilities** in patched functions. It combines rule-based static analysis with deep learning techniques specifically, GraphCodeBERT embeddings and a Variational Autoencoder (VAE) to identify **"silent vulnerabilities"** that persist after fixes due to incomplete patches or overlooked risks. HYDRA operates in an **unsupervised** setting and was evaluated on three real-world projects **Chrome**, **Android**, and **ImageMagick** where it successfully predicted **13.7%**, **20.6%**, and **24%** of patched functions, respectively, as containing potential latent risks. HYDRA outperforms baselines using only regex-based or hybrid symbolic models by surfacing deeply buried but risky patterns strengthening the case for hybrid vulnerability prediction in security audits.<br><br>



## Dataset

The Dataset we used in the paper:

1. **Big-Vul** [[https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing)]  
   A large dataset of known vulnerabilities with vulnerable and fixed code function pairs.<br><br>


## t-SNE Visualization of HYDRA

<p align="center">
  <img src="HYDRA/artifacts/RQ2 and RQ3/t-SNE-and-metrics-score-for-HYDRA/chrome-t-SNE.png" alt="t-SNE visualization of HYDRA embeddings" width="700"/><br>
  <img src="HYDRA/artifacts/RQ2 and RQ3/t-SNE-and-metrics-score-for-HYDRA/android-t-SNE.png" alt="t-SNE visualization of HYDRA embeddings" width="700"/><br>
  <img src="HYDRA/artifacts/RQ2 and RQ3/t-SNE-and-metrics-score-for-HYDRA//imageMagick-t-SNE.png" alt="t-SNE visualization of HYDRA embeddings" width="700"/>
</p><br><br>


## Unsupervised Clustering Metrics Score

<p align="center">
  <img src="HYDRA/artifacts/RQ2 and RQ3/t-SNE-and-metrics-score-for-HYDRA/unsupervised-metrics-score.png" alt="t-SNE visualization of HYDRA embeddings" width="700"/>
</p>


## VAE Reconstruction Loss Curve

<p align="center">
  <img src="HYDRA/artifacts/RQ2 and RQ3/t-SNE-and-metrics-score-for-HYDRA/VAE-loss-curve-HYDRA.png" alt="t-SNE visualization of HYDRA embeddings" width="700"/>
</p>



## References

[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. MSR 2020.