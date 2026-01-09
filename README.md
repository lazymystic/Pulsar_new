# `PULSAR`: Graph based Positive Unlabeled Learning with Multi Stream Adaptive Convolutions for Parkinson’s Disease Recognition


## Overview


![Alt Text](/assets/PULSAR_full_overview.png)


Parkinson's disease (PD) is a neuro-degenerative disorder that affects movement, speech, and coordination. Timely diagnosis and treatment can improve the quality of life for PD patients. However, access to clinical diagnosis is limited in low and middle income countries (LMICs). Therefore, development of automated screening tools for PD can have a huge social impact, particularly in the public health sector. In this paper, we present PULSAR, a novel method to screen for PD from webcam-recorded videos of the finger-tapping task from the Movement Disorder Society - Unified Parkinson’s Disease Rating Scale (MDS-UPDRS). PULSAR is trained and evaluated on data collected from 382 participants (183 self-reported as PD patients). We used an adaptive graph convolutional neural network to dynamically learn the spatio temporal graph edges specific to the finger-tapping task. We enhanced this idea with a multi stream adaptive convolution model to learn features from different modalities of data critical to detect PD, such as relative location of the finger joints, velocity and acceleration of tapping. As the labels of the videos are self-reported, there could be cases of undiagnosed PD in the non-PD labeled samples. We leveraged the idea of Positive Unlabeled (PU) Learning that does not need labeled negative data. Our experiments show clear benefit of modeling the problem in this way. PULSAR achieved 80.95% accuracy in validation set and a mean accuracy of 71.29% (2.49% standard deviation) in independent test, despite being trained with limited amount of data. This is specially promising as labeled data is scarce in health care sector. We hope PULSAR will make PD screening more accessible to everyone. The proposed techniques could be extended for assessment of other movement disorders, such as ataxia, and Huntington’s disease.

[`Link To Paper`](https://zarif98sjs.github.io/PULSAR/)

## Datasets
- [uspark_finger_tapping_train_val](/datasets/uspark_finger_tapping/train_val_data/)
- [uspark_finger_tapping_test](/datasets/uspark_finger_tapping/test_data/)
- [banglapark_finger_tapping_test](/datasets/banglapark_finger_tapping/test_data/)

## Reproducing Test Scores on BanglaPark/USPark
- Clone the repo
- Create a virtual environment
- Install the requirements from requirements.txt

        pip install -r requirements.txt
- Run test_banglapark.py for BanglaPark test set 

        python test_banglapark.py
- Run test_uspark.py for USPark test set

        python test_uspark.py

- A csv file named `banglapark_test_set_scores.csv` or `uspark_test_set_scores.csv` will be created in the root directory
- As mentioned in paper, for each model variant 120 patients will be sampled from the test set and evaluated 20 times. The mean of the 20 scores will be reported in the csv file.


## Training on USPark
- clone the repo
- create a virtual environment
- install the requirements from requirements.txt

        pip install -r requirements.txt
- run train_uspark.py

        python train_uspark.py

- models will be saved in a newly created trained_models folder
- you can change the hyperparameters from the [config file](/config.py)