# MA-Thesis

This repository contains key files regarding my master thesis "[Learning Graph Similarities Using The Weisfeiler-Leman Label Hierarchy](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/22-12-26_SUBMITTED_MA_THESIS_Fabrice_Beaumont.pdf)".
It does NOT contain the used datasets or the computed results (Kernels) on them.
It does NOT contain the entrie evaluation of the proposed method.

## Visual impressions
Method overview:
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20SEMINAR%20TALKS/2022-11-10/images/MasterThesisOverview.jpeg "Title")


Example plot of a *Weisfeiler Lehman Labeling Tree (WLLT)* - with no and learned edge weights:
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20SEMINAR%20TALKS/2022-11-10/images/plot6wllt.png "WLLT example - before edge weight learning")
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20SEMINAR%20TALKS/2022-11-10/images/plot7wllt.png "WLLT example - after edge weight learning")

Example strategy of the evaluation process: Using the resulting accuracy of using a *Support Vector Machine*.
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20THESIS/images/plotA2_SVM_AIDS_GDL_24_17h-05.png "Title")


*Push and Pull*-Experiment on the dataset 'MSRC 9'. Using the learned graph representation (embedding), the graph clusters are visible at initialization (epoch 0). The hyper-parameters allow to soften these clusters, and strenghten them again.

![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20THESIS/images/plotE6_tSNE_e0_MSRC_9_E_GDL_22_00h-05mExp3pull.png "Epoch 0")
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20THESIS/images/plotE6_tSNE_e100_MSRC_9_E_GDL_22_00h-05mExp3pull.png "Epoch 100")
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20THESIS/images/plotE6_tSNE_e400_MSRC_9_E_GDL_22_00h-05mExp3pullpush.png "Epoch 400")

Aim: Improvement over the baseline graph kernel: Partially achieved. For more information read the thesis or contact me directly.
![alt text](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20SEMINAR%20TALKS/2022-11-10/images/2019_Schulz_AccTabelle.png "Title")

(Also considere the seminar talk slides for more information. 
E.g. ![2022-11-10](https://github.com/FabriceBeaumont/MA_INF_MasterThesis/blob/master/0%20SEMINAR%20TALKS/2022-11-10/2022-11-10.pdf))
