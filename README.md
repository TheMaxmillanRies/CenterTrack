# Tracking Objects as Points
By: Maxmillan Ries (5504066), Nafie El Coudi El Amrani (4771338), Cristian Mihai Rosiu (5632226), Jasper Konijn (4383788)
Simultaneous object detection and tracking using center points:

## Introduction

Tracking objects in a given video or series of images is a common and useful computer vision procedure. In "Tracking Objects as Points", Xingyi Zhou uses the CenterNet Object Detection neural network-based algorithm to present a point-based framework for joint detection and tracking, referred to as CenterTrack. In this article, we will reproduce the algorithm described in this paper, and evaluate the effect of different parameters which were described. Specifically, we will train the principle algorithm using different hyperparameters known to affect neural networks, and using parameters unique to the tracking aspect provided by this paper.

## CenterTrack Basics

Xingyi Zhou's algorithm takes as an input several combinations of images. For the principle and most performing algorithm CenterTrack takes 2 timestamp images belonging to a video and a heatmap with the tracking points of the earlier timestamped image, as shown on the left of Figure 1. The tracked object points are obtained using the CenterNet network prediction.


![Basic Input and Output of the CenterTrack network](https://github.com/TheMaxmillanRies/CenterTrack/blob/main/readme/fig2.png)
|:--:|
| <b>Figure 1: Basic Input and Output of the CenterTrack network.</b>|

Using these three images as inputs, the CenterTrack algorithm outputs a set of tracking points for the recent timestamp, a bounding box size map and an offset map.

## Our Setup and Code Replication

The goal of our reproducibility project is to reproduce the results of table 4 in the paper. This table contains the accuracy metrics of an ablation study performed on three different datasets. We limit our project to the MOT17 dataset. In this blog we describe the process of replicating the ablation study experiments as well as the results on some experiments on the effect of different hyperparameters.

The CenterTrack github and the research paper provided clear details into the inner workings of their setup, including the optimization strategy, hyperparameters, and the various loss functions and their combinations. The github provides installation instructions on how to setup the conda environment, clone the repository and download the pretrained models. A seperate page provides instructions on how to obtain the dataset, convert it to the used format and to create the train validation split of the dataset. The github also contains the code that is used to start training or testing of a selection of experiments together with the amount of GPU's used for the experiment, the training time and the test time.

Our project started with the objective of getting the code that was provided in the github running. We were not able to do this on a local system with a Windows or a Mac operating system. Therefore we used the google colab environment to run the project on a ubuntu operating system. With the instructions from the github installation readme and some debugging of problems related to the instalation of packages in the conda environment and a known problem with the logger.py file, we were able to run the code to do tests on the downloaded models. To also run the code for training a model we had to do some additional debugging and found out that we could not run the code for training as provided on the github, due to a lack of memory we had to restrict our batch size to 12. After overcoming these problems we could start training and found out that the google colab environment was too slow to carry out the training. We had to migrate our project to the google cloud platform to enable the use of GPU's.

The authors of the paper used a server setup with 4 TITAN V GPUs (12 GB Memory) for training and a local machine with TITAN Xp GPU for testing. With this hardware they needed 2 hours of training time and 45ms of testing time. We use the google cloud platform to create an instance of a virtual machine in which we are able to recreate the training environment. For this project we have a limitted budget for computing power and therefore we could not copy the setup that the authors used. To make optimal use of our budget we decided on using a virtual machine with a single NVIDIA TESLA T4 (16 GB Memory). This put some restraints on our ability to reproduce the experiments. The maximal batch size that could fit in our memory was 12 instead of 32 used by the original authors. Also our training time was a lot longer with about 13 minutes per epoch. Training the models for the experiments of table 4 took around 14,5 hours per model. Due to these limitations we decided to research the effect of the batch size on the models accuracy and the effect of the amount of training epochs. Further experiments were carried out with an epoch count of 25. This shortened the training time to 5,5 hours and enabled us to run multiple experiments in a day.

We researched the effect of the hyperparameters listed below. Where the installation instructions and documentation on the experiments conducted for the paper was extensive, the documentation of the code itself with comments was not there at all. The structure of the code was very logical with a script for training and a script for testing that each used classes that were contained in seperate files. The choice for variable names was mostly logical but often abriviated. This made it difficult to figure out what arguments for the training and testing script affected what hyperparameters. We compared the code, github instructions on preconfigured experiments, explanation of experiments and results in the paper, and help strings in the argument parser class: opts.py, to guess which arguments to use for our experiments. This part in extending the reproduced code with new experiments is therefore very prone to errors.

- Batch Size
- Epoch Count
- Optimization Functions
- Bounding Box Thresholds
- Heatmap Thresholds
- Combined Thresholds

In the experiments that did not investigate the effect of the hyperparameters listed above, we chose to keep them constant as listed below:

- Batch Size: 12
- Epoch Count: 25
- Bounding Box Threshold: 0.3 (Default)
- Heatmap Threshold: -1 (Default)
- Optimization Function: Adam (LR = 1.25e-4, LR step = 60)

For each of our experiments, we trained two networks, both networks were trained with identical parameters, with the only key distinction being that the first of the two networks used a downloaded model that was pretrained on the crowdhuman dataset (referred to as "crowdhuman" in the CenterTrack paper).
By doing this, we generally aimed to investigate if retuning the network trained on the crowdhuman would lead to similar paterns in the results compared to training a new network only on the MOT17 dataset.
As our code was run on the google cloud/collab platform, the following link shows the majority of steps required for setting up the collab aspect: https://colab.research.google.com/drive/1p3QKDwT239xolY7_MsaGszlpZmPUncb0?usp=sharing

----

### Batch Size
As we started our training experiments, we tried staying as close to the original setup parameters as possible. However, while the paper clearly states that 4 Titan Xp 12GB VRAM Graphics Card was used, a T4 with 16GB of RAM was found to be far lacking. As a consequence, we could use at most a batch of 12, in comparison to the paper's stated batch size of 32. 

Taking the lacking batch size as a source of inspiration, we retrained the basic CenterTrack network on varying batch sizes, resulting in the table below. All parameters were kept constant relative to the basic MOT half train/val split described in the paper, with only the epoch count being decreased though kept constant. 
<br />
<table align="center">
  <tr>
    <th>Batch Size</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <th>Batch Size</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>1</td>
    <td>54.4%</td>
    <td>4.1%</td>
    <td>40.6%</td>
    <td>1.2%</td>
  </tr>
  <tr>
    <td>2</td>
    <td>59.0%</td>
    <td>6.9%</td>
    <td>33.0%</td>
    <td>1.1%</td>
  </tr>
  <tr>
    <td>4</td>
    <td>59.7%</td>
    <td>9.4%</td>
    <td>29.9%</td>
    <td>1.0%</td>
  </tr>
  <tr>
    <td>8</td>
    <td>63.8%</td>
    <td>7.4%</td>
    <td>27.9%</td>
    <td>1.2%</td>
  </tr>
  <tr>
    <td>12</td>
    <td>65.8%</td>
    <td>4.1%</td>
    <td>29.2%</td>
    <td>1.0%</td>
  </tr>
</table>
<div align="center"> Table 1: Accuracies of FP, FN, IDSW and MOTA over several batch sizes.</div>
<br />


![Line chart showing the evolution of FP, FN, IDSW and MOTA over time](https://github.com/TheMaxmillanRies/CenterTrack/blob/main/readme/batch_lines.png)
|:--:|
| <b>Figure 2: Line chart showing the evolution of FP, FN, IDSW and MOTA over time. Overall, the lines are following a constant trend, with the FN and FP features presenting some small spikes in the middle and close to the end</b>|

Looking at the table above, the increase in the batch sizes seems to strongly correlate with an increase in the MOTA (Multiple Object Tracking Accuracy) and a general decreasein the FN count (False Negative). As a larger batch sizes allows a better estimation of the gradient, the general improvement in the accuracy seems fair and accurate to theoretical expectations.

The IDSW percentage (When objects are successfully detected but not tracked) generally remains constant, leading us to assume that the tracking is consistent, and the detection is lacking.

----

### Epoch Count
One of the observed difficulties when reproducing the CenterTrack paper was the time required to train. For a single epoch to train on a batch size of 12, it would take 12min24s on average. As the paper states to train for 70 epochs, the total training time for a single model would be ~14.5 hours.

To investigate the effect of epoch size on the accuracy of CenterTrack, we trained the base model with different epoch counts, whilst maintaining the same parameters and train/val split as described in the paper. The results using a pre-trained model are shown below:

<br />
<table align="center">
  <tr>
    <th>Epoch Count</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>5</td>
    <td>62.4%</td>
    <td>4.6%</td>
    <td>30.8%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>63.5%</td>
    <td>3.4%</td>
    <td>30.8%</td>
    <td>2.3%</td>
  </tr>
  <tr>
    <td>15</td>
    <td>64.1%</td>
    <td>2.7%</td>
    <td>31.1%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>64.6%</td>
    <td>4.4%</td>
    <td>28.9%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>25</td>
    <td>64.8%</td>
    <td>3.6%</td>
    <td>29.6%</td>
    <td>1.9%</td>
  </tr>
  <tr>
    <td>30</td>
    <td>64.7%</td>
    <td>5.4%</td>
    <td>28.0%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>35</td>
    <td>63.4%</td>
    <td>4.0%</td>
    <td>30.6%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>40</td>
    <td>63.6%</td>
    <td>4.7%</td>
    <td>29.6%</td>
    <td>2.1%</td>
  </tr>
</table>
<div align="center"> Table 2: Accuracies of FP, FN, IDSW and MOTA over different epoch counts on the crowdhuman pretrained network.</div>
<br />

The final results shows a very constant trend over time. We can clearly see that a pre-trained model is not necessarily helping too much with the classification. Most of the metrics stay flat throughout the training.

![Line chart showing the evolution of FP, FN, IDSW and MOTA over time.](https://github.com/TheMaxmillanRies/CenterTrack/blob/main/readme/epoch_lines.png)
|:--:|
| <b>Figure 3: Line chart showing the evolution of FP, FN, IDSW and MOTA over time using a pre-trained model. Overall, the lines are following a constant trend, with the FN and FP features presenting some small spikes in the middle and close to the end</b>|

The next step was to see if using a fresh model would improve the performance. Therefore, we ran again the same experiment but on a model that doesn't have any prior knowledge. The results can be seen in the following table.

<br />
<table align="center">
  <tr>
    <th>Epoch Count</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>5</td>
    <td>52.8%</td>
    <td>9.3%</td>
    <td>35.9%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>55.5%</td>
    <td>7.4%</td>
    <td>34.9%</td>
    <td>2.2%</td>
  </tr>
  <tr>
    <td>15</td>
    <td>57.0%</td>
    <td>6.7%</td>
    <td>34.2%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>54.8%</td>
    <td>8.8%</td>
    <td>34.2%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>25</td>
    <td>56.9%</td>
    <td>7.0%</td>
    <td>34.0%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>30</td>
    <td>56.5%</td>
    <td>7.6%</td>
    <td>33.8%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>35</td>
    <td>57.0%</td>
    <td>7.1%</td>
    <td>33.9%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>40</td>
    <td>57.7%</td>
    <td>6.4%</td>
    <td>33.0%</td>
    <td>2.0%</td>
  </tr>
</table>
<div align="center"> Table 3: Accuracies of FP, FN, IDSW and MOTA over different epoch counts.</div>
<br />

Without the pre-trained model, the accuracy went down. However, there is a slight change in the shape of accuracy over time. Looking at Figure 4, one can observe that the lines are now showing better upward (e.g. MOTA) and downward (e.g. FP) trends. This could indicate that, given more time and more epochs, the model could achieve the stated results. However, there is still an enormous amount of uncertainty and we can't draw a clear conclusion from only looking at these two graphs.

![Line chart showing the evolution of FP, FN, IDSW and MOTA over time.](https://github.com/TheMaxmillanRies/CenterTrack/blob/main/readme/epoch_lines2.png)
|:--:|
| <b>Figure 4: Line chart showing the evolution of FP, FN, IDSW and MOTA over time without a pre-trained model. Overall, the lines look very similar, however, we can clearly see a more upward trend for MOTA and a downwards trend for False Positive compared to the previous experiment</b>|

----
### Optimization Functions
One of the assumed parameters of the CenterTrack is the usage of the Adam optimization function with a learning rate of 1.25e - 4. This specific choice of optimization function is left unmentioned in the paper, and we thought it interesting to evaluate what we recently learned in the Deep Learning course, and tried training the CenterTrack network on Momentum and RMSProp.
The expectation with the training is for Adam to perfom the best, with RMSProp coming in second and Momentum in third.
<br />
<table align="center">
  <tr>
    <th>Optimizer</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>Momentum</td>
    <td>61.1%</td>
    <td>8.7%</td>
    <td>27.7%</td>
    <td>2.5%</td>
  </tr>
  <tr>
    <td>RMSProp</td>
    <td>64.3%</td>
    <td>6.0%</td>
    <td>27.1%</td>
    <td>2.3%</td>
  </tr>
  <tr>
    <td>Adam</td>
    <td>64.1%</td>
    <td>4.5%</td>
    <td>29.4%</td>
    <td>1.9%</td>
  </tr>
</table>
<div align="center"> Table 4: Accuracies of FP, FN, IDSW and MOTA using Momentum, RMSProp and Adam on the crowdhuman pretrained network.</div>
<br />

The results of our experiment on the pretrained network indicate that the RMSProp optimization function best trained the network. While these results could not be verified with multiple runs (due to time constraints), this observation feels false. Firstly, the Adam optimization function builds on RMSProp, and combines it with Momentum to create a more robust function. 

Additionally, the IDSW column indicates, that the Adam trained network most successfully tracked an object if it was detected (difference of at least 0.4). The FP count is also lower for Adam, leading us to believe that the models trained with RMSProp and Momentum made more incorrect detections, while the model trained by Adam made more conservative detections. This would explain why the RMSProp results appear better than that of Adams, though the 
difference is negiligible enough to not be considered significant.

<br />
<table align="center">
  <tr>
    <th>Optimizer</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>Momentum</td>
    <td>45.9%</td>
    <td>13.7%%</td>
    <td>37.9%%</td>
    <td>2.4%</td>
  </tr>
  <tr>
    <td>RMSProp</td>
    <td>56.6%%</td>
    <td>7.0%</td>
    <td>33.9%</td>
    <td>2.5%</td>
  </tr>
  <tr>
    <td>Adam</td>
    <td>54.7%</td>
    <td>8.9%</td>
    <td>33.9%</td>
    <td>2.5%</td>
  </tr>
</table>
<div align="center"> Table 4: Accuracies of FP, FN, IDSW and MOTA using Momentum, RMSProp and Adam on the freshly trained network.</div>
<br />

On the freshly trained network (shown above), the results show a similar view, with the RMSprop network outperforming the Adam trained network. This unfortunately seems to disprove our initial theory of falsely tested results. The Adam network is outperformed by the RMSprop network across every score, leading us to believe that the epoch count (limited to 25 on freshly trained networks for time constraint reasons) might have been the determining factor. It is still however possible that the RMSprop optimization function is simply better suited to train the CenterTrack project.

----

### Bounding Box Threshold
The paper vaguely describes the θ thresholds as the "bounding box" confidence threshold. It does however describe that the MOTA is sensitive to the task-dependent output threshold θ. Additionally, the paper describes an optimal combination of threshold of θ = 0.5 and 𝜏 = 0.4 (see Heatmap Threshold section for 𝜏). In order to further learn about the influence of this parameter, we decided to investigate its effect on CenterTrack's performance.

The CenterTrack project comes with a series of optional parameters which can be set and modified manually for any training. After some reading into the various parameters available, we found that the --track_-_tresh parameter translated to the θ threshold.

<br />
<table align="center">
  <tr>
    <th>θ</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>64.7%</td>
    <td>5.9%</td>
    <td>27.6%</td>
    <td>1.9%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>63.3%</td>
    <td>10.1%</td>
    <td>25.2%</td>
    <td>1.4%</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>63.0%</td>
    <td>7.6%</td>
    <td>27.7%</td>
    <td>1.7%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>63.2%</td>
    <td>6.7%</td>
    <td>27.9%</td>
    <td>2.2%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>62.6%</td>
    <td>4.6%</td>
    <td>30.3%</td>
    <td>2.5%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>59.2%</td>
    <td>1.5%</td>
    <td>37.0%</td>
    <td>2.3%</td>
  </tr>
</table>
<div align="center"> Table 5: Accuracies of FP, FN, IDSW and MOTA with a pre trained data using different values for the Bounding Box Threshold on the crowdhuman pretrained network.</div>
<br />

Observing the results of the experiment, one can clearly see that increasing the θ threshold results in a general decrease of the MOTA and the FP percentage, and an increase in both the FN and IDSW results. We see generally that higher bounding box threshold causes a decrease in performance.  

Since the previous results are run with a pretrained data, we opted to run the same experiments on fresh data to see whether the pretrained data makes a significant difference between the two experiments. We present the results in the following table:

<br />
<table align="center">
  <tr>
    <th>θ</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>56.0%</td>
    <td>10.4%</td>
    <td>31.5%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>55.1%</td>
    <td>11.6%</td>
    <td>31.9%</td>
    <td>1.5%</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>53.8%</td>
    <td>11.8%</td>
    <td>32.6%</td>
    <td>1.8%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>55.1%</td>
    <td>6.8%</td>
    <td>35.7%</td>
    <td>2.5%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>56.3%</td>
    <td>4.9%</td>
    <td>36.6%</td>
    <td>2.2%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>51.6%</td>
    <td>1.3%</td>
    <td>44.1%</td>
    <td>3.0%</td>
  </tr>
</table>
<div align="center"> Table 6: Accuracies of FP, FN, IDSW and MOTA with a fresh data using different values for the Bounding Box Threshold.</div>
<br />

Although the results are slightly worse than the results achieved using the pretrained data, they follow generally the same trends. As shown in the table, the MOT Accuracy (MOTA) drops by almost 5% and that the FP values are decreasing as the Bounding Box Threshold increases. 

This decrease in both sets of results contradicts what was mentioned on the paper and can be explained with the difference in settings between the experiments on the paper and the settings we set up for ourselves. First, the bath size during these experiments was 8 instead of 32 as the authors did in the paper and low epoch counts.

----

### Heatmap Threshold
In order to improve the tracking capabilities of CenterTrack, Xingyi Zhou introduced the usage of prior detections as an additional input. Using the point-based nature of the tracking to his/her advantage, CenterTrack renders all detections in a class-agnostic single-channel heatmap using a Gaussian render function. To avoid propagation of false positive detections, only the objects with a certain confidence score greater than threshold 𝜏 are rendered. This 𝜏 hyperparameters is specifically set by the user before training.

We chose to investigate the benefit of this parameter vis-a-vis the false positive propagation and general accuracy. In the CenterTrack implementation, this is alterable using the pre-thresh optional argument upon training and testing.

<br />
<table align="center">
  <tr>
    <th>𝜏</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>64.7%</td>
    <td>5.9%</td>
    <td>27.6%</td>
    <td>1.9%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>63.4%</td>
    <td>6.1%</td>
    <td>28.5%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>62.9%</td>
    <td>4.2%</td>
    <td>30.6%</td>
    <td>2.4%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>65.5%</td>
    <td>4.4%</td>
    <td>28.2%</td>
    <td>1.8%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>63.3%</td>
    <td>4.7%</td>
    <td>29.3%</td>
    <td>2.4%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>65.1%</td>
    <td>5.5%</td>
    <td>27.5%</td>
    <td>1.9%</td>
  </tr>
</table>
<div align="center"> Table 7: Accuracies of FP, FN, IDSW and MOTA with pre-trained data using different values for the Heatmap Threshold on the crowdhuman pretrained network.</div>
<br />

Observing the table, we could not find a specific pattern which suggests that the 𝜏 thresholds truly reduces the propagation of false positive detections. The same observation can be seen in the number changes of the MOTA and false negatives. The results of this group of experiments can also be explained by the difference in epoch count and small batch sizes. 

The next set of experiments were run on fresh data that is not pretrained . The results are first presented and then a small analysis of data is done. 

<br />
<table align="center">
  <tr>
    <th>𝜏</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>56.0%</td>
    <td>10.4%</td>
    <td>31.5%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>55.5%</td>
    <td>7.5%</td>
    <td>34.7%</td>
    <td>2.7%</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>57.3%</td>
    <td>6.5%</td>
    <td>33.9%</td>
    <td>2.3%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>56.2%</td>
    <td>8.3%</td>
    <td>32.7%</td>
    <td>2.8%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>56.4%</td>
    <td>6.4%</td>
    <td>34.9%</td>
    <td>2.4%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>56.6%</td>
    <td>7.1%</td>
    <td>34.0%</td>
    <td>2.3%</td>
  </tr>
</table>
<div align="center"> Table 8: Accuracies of FP, FN, IDSW and MOTA with fresh data using different values for the Heatmap Threshold.</div>
<br />

Similar to the results of the Bounding Box Threshold, the general results are slightly worse than the ones we got with the pretrained  data. We, generally, observed a slight increase in accuracy with bigger threshold value. However, the values of FPs and FNs were not following any trend. 

This slight decrease is primarily due to the fact that the data used in these experiments is not pretrained. Therefore, we conclude that using pretrained data for CenterTrack results in better results. 


As we couldn't run the experiments with higher epoch count and batch sizes, we can't say for sure that this trend of semi-random values will happen again in another experiment setting. Therefore, we conclude that these experiments are not representative of the performance of CenterTrack and need more investigation to be able to find an optimal value and reach numbers similar to the ones reported on the paper. 

----

### Combined Thresholds
In the previous sections, we elaborated on our experiments aimed at testing the influence of the bounding box and heatmap thresholds separately. The CenterTrack paper however clearly states that for the MOT17 dataset, a θ = 0.5 and 𝜏 = 0.4 combination is optimal according to their experiments.

Due to our limited timeframe, we investigated several combinations of θ and 𝜏, where both variables were equal, as depicted below:

<br />
<table align="center">
  <tr>
    <th>θ</th>
    <th>𝜏</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>0.0</td>
    <td>43.5%</td>
    <td>31.5%</td>
    <td>24.0%</td>
    <td>1.0%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>0.1</td>
    <td>64.7%</td>
    <td>7.8%</td>
    <td>26.3%</td>
    <td>1.2%</td>
  </tr>
  <tr>
    <td>0.2</td> 
    <td>0.2</td>
    <td>65.9%</td>
    <td>6.3%</td>
    <td>26.4%</td>
    <td>1.3%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>0.3</td>
    <td>64.0%</td>
    <td>5.1%</td>
    <td>29.0%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>0.4</td>
    <td>62.2%</td>
    <td>5.1%</td>
    <td>29.0%</td>
    <td>2.0%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>0.5</td>
    <td>58.2%</td>
    <td>1.8%</td>
    <td>36.8%</td>
    <td>3.2%</td>
  </tr>
</table>
<div align="center"> Table 9: Accuracies of FP, FN, IDSW and MOTA with pre-trained data using different values for the Heatmap Threshold on the crowdhuman pretrained network.</div>
<br />

Similar to the previous set of experiments, no clear trend can be observed in these test except a decrease in false positive as the values of both θ and 𝜏 get higher. This observation can probably be explained with the . The values seem to be optimal at θ = 0.2 and 𝜏 = 0.2. However, the decision to run the experiments with same values for both parameters was arbitrary. Therefore, we cannot say anything about the results of the experiments where both parameters are different (possible tests to run in the future to verify the claims of the paper). 

In the following table, we run exactly the same experiment with fresh data. 

<br />
<table align="center">
  <tr>
    <th>θ</th>
    <th>𝜏</th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>0.0</td>
    <td>56.0%</td>
    <td>10.4%</td>
    <td>31.5%</td>
    <td>2.1%</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>0.1</td>
    <td>54.9%</td>
    <td>11.1%</td>
    <td>32.7%</td>
    <td>1.3%</td>
  </tr>
  <tr>
    <td>0.2</td> 
    <td>0.2</td>
    <td>57.2%</td>
    <td>7.7%</td>
    <td>33.5%</td>
    <td>1.6%</td>
  </tr>
  <tr>
    <td>0.3</td>
    <td>0.3</td>
    <td>63.1%</td>
    <td>5.1%</td>
    <td>29.6%</td>
    <td>2.2%</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>0.4</td>
    <td>63.4%</td>
    <td>3.4%</td>
    <td>30.9%</td>
    <td>2.3%</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>0.5</td>
    <td>58.5%</td>
    <td>1.7%</td>
    <td>36.7%</td>
    <td>3.2%</td>
  </tr>
</table>
<div align="center"> Table 10: Accuracies of FP, FN, IDSW and MOTA with fresh data using different values for the Heatmap Threshold.</div>
<br />

Table 2 shows the influence of both the bounding box and heatmap thresholds on the model's accuracy.
As one can observe, increasing the combined thresholds results in a similar pattern on both the pretrained and freshly trained networks.
Up until thresholds of 0.4, the network shows a continuous and regular improvement in performance. 
However, as the threshold is increased further, a negative trend reveals itself. This is likely caused by the network no longer using many heatmaps as inputs, as even somewhat confident ones are discarded.

----

### Table 4 - MOT17
As instructed as part of the reproduction of the code, we investigate the results of Table 4 and attempted to reproduce them. While the exact results could not be perfectly reproduce, the method used to obtain each experiment was verified in conjunction with the mathematical notions denoted in the paper.

<br />
<table align="center">
  <tr>
    <th></th>
    <th>MOTA</th>
    <th>FP</th>
    <th>FN</th>
    <th>IDSW</th>
  </tr>
  <tr>
    <td> w/o offset</td>
    <td>63.8%</td>
    <td>4.3%</td>
    <td>30.0%</td>
    <td>1.9%</td>
  </tr>
  <tr>
    <td>w/o heatmap</td>
    <td>65.3%</td>
    <td>3.8%</td>
    <td>29.0%</td>
    <td>1.8%</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>66.1%</td>
    <td>4.5%</td>
    <td>28.4%</td>
    <td>1.0%</td>
  </tr>
</table>
<div align="center"> Table 11: Reproduction of Table 4 in the original paper.</div>
<br />

The principle difference with the results of Table 4 from the paper is that removing the offset is seen to be far more detrimental. While the FP count is similar, the FN and IDSW counts are higher than the paper's counterpart and the MOTA is substantially lower. To be perfectly honest, we are not sure why this is specifically the case. The paper does not properly describe how the experiments were specifically conducted for Table 4, beyond providing a general description of what each experiment aimed to investigate.

As we made sure to use the validation split of section 5.1, our suspicion is that the model were fully trained without using transfer-learning, with the crowdhuman provided model corresponding to the most successful training (the "Ours" training result).
Unfortunately, we did not have the time to investigate the possibility, as fine-tuning the pretrained network takes ~14.5 hours, and training the entire network from scratch would take over a day.

----

## Conclusion
The CenterTrack paper presents itself as a fresh attempt at object detection and tracking based on a point-based system. While some of our results contradict the statements presented in the paper, notably with regards to the thresholds, the majority of our results follow the same interesting trends. For now, as we have not had the chance to evaluate the paper's results on a different dataset, we can only remain skeptical about the performance of the paper outside of it's testing domain. The author does however describe the project to work on video data, and the KTTI dataset, providing some confidence in its applicability. 

The only personal fault with the paper's results is the final optimal 66.1% accuracy, which generally makes this project harder to use in an interesting context, as its results feel unreliable (slightly better than a coin flip). We feel that the error propagation caused by the hierarchical structure (Detection -> Tracking) causes the mixed results, as the IDSW results consistently indicate that the Tracking is accurate, but the CenterNet detection is not. An investigation of another point-based detection network within the CenterTrack context could be very interesting.
