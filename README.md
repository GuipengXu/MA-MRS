# DA-MRS

This repository contains our implementations for Denoising and Aligning Multi-modal Recommender System. We provide dataset processing methods, detailed implementation methods, and running logs for all experiments. 



### Folder structure

- data: Recommendation datasets used in our experiments.
- src: The implementation of DA-MRS and its baselines on Amazon Baby, Sports and Clothing. 
- src_ablation_log:  The running logs for the ablation studies.
- src_cold_start: The Implementation of DA-MRS on the cold-start user testing experiment. 
- src_parameter: The running logs for the parameter experiments.
- src_tiktok: The implementation of DA-MRS+LightGCN and its baselines on TikTok dataset. 
- src_visualization: The visualization of DA-MRS. 



### Dependencies

- OS: Ubuntu 20.04.6 LTS
- numpy==1.23.5
- pandas==1.5.3
- python==3.10.9
- scipy==1.10.0
- torch==2.0.0
- pyyaml==6.0
- pyg==2.3.0
- networkx==2.8.4



### Main Experiment

1. Step 1: Run the following commend to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=baby --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.
   2. topk: int type, parameter for pruning the Item-item behavior graph.

   

2. Step 2: Specify configurations

   1. Go to the src folder
   
      ```bash
      # go to the src folder
      cd src
      ```
   
2. Specify dataset-specific configurations
   
   ```bash
   vim configs/dataset/xx.yaml
   ```
   
   3. Specify model-specific configurations
   
      ```bash
      vim configs/model/xx.yaml
      ```
   
   4. Specify the overall configurations
   
      ```bash
      vim configs/overall.yaml
      ```
   
3. Step 3: Run the following commend to train and evaluate the model. 

   ```python
   # run the code 
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the **backbone model**, such as LightGCN. 
   
   2. dataset: str type, the name of the dataset, such as baby, sports, clothing.
   
   3. gpu_id: str type, the specified GPU. 
   
      

**We provide DA-MRS+LightGCN, DA-MRS+MFBPR, DA-MRS+VBPR, and all baselines on Amazon Baby, Sports, and Clothing datasets in the log/ folder for reference.**



### Visualization

***\*The visualization results are in the “./src_visualization/image” folder.\****

We produce two interpretative images.

1. The first image visualizes the item representations before (i.e., using LightGCN) and after denoising and aligning (i.e., using DA-MRS+LightGCN). Specifically, we project the learned item representations to 2-dimensional normalized vectors on a unit sphere (i.e., a circle with radius 1) by using t-SNE. All the representations are obtained when the methods reach their best performance.
2. Since all the projected representations are on the sphere, they only differ in the polar angle in a polar coordinate system. The second image plots the distribution of polar angles with the nonparametric Gaussian kernel density estimation [1].

As shown in the Figures:  

1. The first image illustrates item representations obtained from LightGCN **are clustered on the circle** (i.e., a few segments on the circle include many points while other segments include a few points). Item representations from DA-MRS are **more evenly distributed on the sphere**. 

2. The second image illustrates that the polar angles before denoising mainly reside in some regions (i.e., the distribution has several peaks), and **the density distribution of polar angles is smoother after denoising and aligning**.

A more uniform representation distribution can improve the generalization ability [1]. This suggests the model can learn more universally effective item representations through DA-MRS. As the ablation study in Section 4.4 shows, DA-MRS achieves better recommendation performance.



We will add the visualization on the Baby, Sports, and Clothing datasets in the revision.



[1] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. 2022. Are Graph Augmentations Necessary? Simple Graph Contrastive&nbsp;Learning for Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, 1294–1303.

![Baby](./src_visualization/image/baby_cmp_distribution.jpg)

Figure 1: The visualization experiment on baby dataset. 



![sports](./src_visualization/image/sports_cmp_distribution.jpg)

Figure 2: The visualization experiment on sports dataset. 



![Clothing](./src_visualization/image/clothing_cmp_distribution.jpg)

Figure 3: The visualization experiment on clothing dataset. 



### Cold Start Experiment

1. Step 1: Run the following commend to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=baby --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.

   2. topk: int type, parameter for pruning the Item-item behavior graph.

      

2. Step 2: Specify configurations

   1. Go to the src_cold_start folder

      ```bash
      # go to the src folder
      cd src_cold_start
      ```

   2. Specify dataset-specific configurations

      ```bash
      vim configs/dataset/xx.yaml
      ```

   3. Specify model-specific configurations

      ```bash
      vim configs/model/xx.yaml
      ```

   4. Specify the overall configurations

      ```bash
      vim configs/overall.yaml
      ```

   5. Differentiating user groups

      **In utils/dataset.py , lines 70-74**, we can differentiat user groups. 

   ```python
    a = self.df['userID'].value_counts()
    # set the active user group
    warm = a[a>=50].index.tolist()
   
    # set the new user group 
    cold = a[a==5].index.tolist()
    dfs.append(dfs[2][dfs[2]['userID'].isin(warm)])
    dfs.append(dfs[2][dfs[2]['userID'].isin(cold)])
   ```

3. Step 3: Run the following commend to train and evaluate the model. 

   ```python
   # run the code 
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the backbone model, such as LightGCN and MF.

   2. dataset: str type, the name of the dataset, such as baby, sports, clothing, and tiktok.

   3. gpu_id: str type, the specified GPU. 

      

4. **We provide the logs of DA-MRS+LightGCN and LightGCN in the log/ folder for reference.** 

   - DA-MRS+LightGCN-baby-Dec-10-2023-17-46-27.log: The results of active user group and less active user group. 
   - DA-MRS+LightGCN-baby-Dec-10-2023-18-18-27.log: The results of active user group and new user group. 
   - LightGCN-baby-Dec-10-2023-16-57-36.log: The results of active user group and new user group. 
   - LightGCN-baby-Dec-10-2023-18-20-02.log: The results of active user group and less active user group. 



### TikTok Dataset

1. Step 1: Run the following commend to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=tiktok --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.

   2. topk: int type, parameter for pruning the Item-item behavior graph.

      

2. Step 2: Specify configurations

   1. Go to the src_tiktok folder

      ```bash
      # go to the src_tiktok folder
      cd src_tiktok 
      ```

   2. Specify dataset-specific configurations

      ```bash
      vim configs/dataset/xx.yaml
      ```

   3. Specify model-specific configurations

      ```bash
      vim configs/model/xx.yaml
      ```

   4. Specify the overall configurations

      ```bash
      vim configs/overall.yaml
      ```

      

3. Step 3: Run the following commend to train and evaluate the model. 

   ```python
   # run the code 
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the backbone model, such as LightGCN. 

   2. dataset: str type, the name of the dataset, such as baby, sports, clothing, and tiktok.

   3. gpu_id: str type, the specified GPU. 

      

4. **We provide the logs of DA-MRS+LightGCN and other baselines in the log/ folder for reference.** 



### Noisy scenarios experiments

1. Step 1: Run the following commend to construct item-item behavior graph. 

   ```python
   # construct item-item behavior graph. 
   
   python -u build_iib_graph.py --dataset=baby --topk=2
   ```

   There are two parameters:

   1. dataset: str type, allowed values are baby, sports, clothing, and tiktok.
   2. topk: int type, parameter for pruning the Item-item behavior graph.

   

2. Step 2: Run the following commend to construct noisy scenarios. 

   ```python
   # construct noisy modality scenarios. 
   python -u replace_modality.py
   
   # construct noisy modality scenarios. 
   
   # add noisy feedback
   python -u add_neg_inter.py
   
   # remove noisy feedback
   python -u delete_inter.py
   ```

3. Step 3: Specify configurations

   1. Go to the src folder

      ```bash
      # go to the src folder
      cd src
      ```

   2. Specify dataset-specific configurations

      ```bash
      vim configs/dataset/xx.yaml  Lines 9-13
      ```

      ```python
      inter_file_name: 'baby.inter' # noisy feedback scenarios
      
      # name of features
      vision_feature_file: 'image_feat.npy' # noisy visual content scenarios
      text_feature_file: 'text_feat.npy' # noisy textual content scenarios
      ```

   3. Specify model-specific configurations

      ```bash
      vim configs/model/xx.yaml
      ```

   4. Specify the overall configurations

      ```bash
      vim configs/overall.yaml
      ```

4. Step 3: Run the following commend to train and evaluate the model. 

   ```python
   # run the code 
   python -u main.py --model=LightGCN --dataset=baby --gpu_id=0
   ```

   There are three parameters: 

   1. model: str type, the name of the **backbone model**, such as LightGCN. 

   2. dataset: str type, the name of the dataset, such as baby, sports, clothing.

   3. gpu_id: str type, the specified GPU. 

      


### Ablation Study and Parameter Study

**By modifying the corresponding model parameters, experiments can be conducted. The experimental steps are consistent with the main experiment. We provide the running logs of all experiments for reference.**

- src_ablation_log: Running log of the **ablation studies** on the Baby, Sports and Clothing dataset. 
- src_ablation_log/CL_experiment:  Running log of the experiments which investigates the effects of **different strategies to select positive and negative samples** on the Baby dataset. 
- src_parameter_log/IIB_experiment: Running log of the experiments which investigates the effects of **different interaction deletion threshold** on the Baby dataset. 
- src_parameter_log/KNN: Running log of the experiments in Appendix A.4.1, which investigates the effects of the **number of k in DIIG.** 
- src_parameter_log/Loss_ai: Running log of the experiments in Appendix A.4.3, which investigates the effects of **weight of** $L_{AI}$. 
- src_parameter_log/Loss_au: Running log of the experiments in Appendix A.4.2, which investigates the effects of **weight of** $L_{AU}$. 





### Acknowledgement

The datasets and the structure of this code are based on [MMRec](https://github.com/enoche/MMRec). Thanks for their work.