# DA-MRS



### Visualization experiments. 

**The visualization results are in the “./image” folder.**

We visualize the item representations before (i.e., using LightGCN) and after denoising (i.e., using DA-MRS+LightGCN). Specifically, we first project the learned item representations to 2-dimensional normalized vectors on a unit sphere (i.e., a circle with radius 1) by using t-SNE. All the representations are obtained when the methods reach their best performance. Since all the projected representations are on the sphere, they only differ in the polar angle in a polar coordinate system. Then, we plot the distribution of polar angles with the nonparametric Gaussian kernel density estimation [1]. 

As shown in the Figures, item representations obtained from LightGCN are clustered, and the polar angles mainly reside in some regions. After denoising, item representations from DA-MRS are more evenly distributed on the sphere, and the density distribution of polar angles is more smooth. A more uniform representation distribution can improve the generalization ability [1]. This suggests that the model can learn more universally effective item representations through DA-MRS. As the ablation study in Section 4.4 shows, DA-MRS achieves better recommendation performance.

![Baby](./image/baby_cmp_distribution.jpg)

Figure 1: The visualization experiment on baby dataset. 

![sports](./image/sports_cmp_distribution.jpg)

Figure 2: The visualization experiment on sports dataset. 

![Clothing](./image/clothing_cmp_distribution.jpg)

Figure 3: The visualization experiment on clothing dataset. 



[1] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are Graph Augmentations Necessary? Simple Graph Contrastive&nbsp;Learning for Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, 1294–1303.

