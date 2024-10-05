# COBRA - COnfidence score Based on shape Regression Analysis for method-independent quality assessment of object pose estimation from single images

**COBRA - COnfidence score Based on shape Regression Analysis for method-independent quality assessment of object pose estimation from single images** 

[Panagiotis Sapoutzoglou<sup>1,2</sup>](https://www.linkedin.com/in/panagiotis-sapoutzoglou-66984a201/), [Georgios Giapitzakis Tzintanos<sup>1</sup>](https://github.com/giorgosgiapis), [George
Terzakis<sup>2</sup>](https://github.com/terzakig), [Maria Pateraki<sup>1,2</sup>](http://www.mpateraki.org/)

[<sup>1</sup>National Technical University of Athens](https://ntua.gr/en/), Athens, Greece <br>
[<sup>2</sup>Institute of Communication & Computer Systems](https://www.iccs.gr/en/), Athens, Greece

[![arXiv link](https://img.shields.io/badge/arXiv-2404.16471-B31B1B.svg)](https://arxiv.org/abs/2404.16471)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Visit%20Site-brightgreen?style=for-the-badge&logo=github)](https://pansap99.github.io/COBRAv1.2/)

**Abstract**: We present a generic algorithm for scoring pose estimation methods that rely on single image semantic analysis. The algorithm employs a lightweight putative shape representation using a combination of multiple Gaussian Processes. Each Gaussian Process (GP) yields distance normal distributions from multiple reference points in the object’s coordinate system to its surface, thus providing a geometric evaluation framework for scoring predicted poses. Our confidence measure comprises the average mixture probability of pixel back-projections onto the shape template. In the reported experiments, we compare the accuracy of our GP based representation of objects versus the actual geometric models and demonstrate the ability of our method to capture the influence of outliers as opposed to the corresponding intrinsic measures that ship with the segmentation and pose estimation methods.

## Overview

The core functionality of this repo can be summarized in six steps:

- Installation: Set up the Conda environment and install dependencies using the provided instructions.
- Sample points from a 3D model to serve as the training and test sets. This is done by utilizing the script ```preprocessing.py```.
- Train COBRA to represent the shape of the object. This is done by running the ```train.py``` script.
- Evaluate the trained model over the query test points with ```infer.py```.
- Compute evaluation metrics with ```eval.py```.
- Use the trained model to score estimated poses pre-computed from an independent pose estimation algorithm with ```score_poses.py```.

## Installation

- Clone the repository and setup the conda environment:
```
git clone https://github.com/pansap99/COBRA.git
cd COBRA
conda env create -f environment.yml
```
- Install the pose visualization toolkit by downloading the wheel file inside the ```vis``` directory:

```
cd vis
pip install pose_vis-1.0-py3-none-any.whl
```
This package utilizes OpenGL to render the estimated poses and overlay them into the images, together with their derived confidence.

**NOTE**: If you encounter an error similar to this one: ```libGL error: MESA-LOADER: failed to open iris``` you can try to resolve it by running :
```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6```

## File organization

You have to place your 3D models under ```./data/models/original/{your_class_name}```. You can change the paths to save the data in ```common.py```.


## Sampling - Preprocessing

We define virtual cameras around the object and ray-casting to acquire points lying only on the outer surface of the object. To sample points for the train and test point-clouds run:

```
python preprocessing.py \
--class-name your_class_name \ 
--num-samples 10000 250000 \ #samples for train and test pcds
```

After runing the command, the folders ```./data/models/train/{your_class_name}``` and ```./data/models/test/{your_class_name}``` will be created containing the train and test point clouds for your models.

## Training 

To train COBRA to represent the shape of the objects you simply run:

```
python train.py \
--class-name your_class_name \
--init-lr 0.1
--cluster-overlap 0.2 
--normalize \ 
--num-steps \ 
--min-num-classes \ 
--max-num-classes \ 
--step
```
A full list of commands can be seen running ```python train.py --help``` as we utilize Tyro cli for the arguments.

After training the trained GP models are saved in ```./data/results/{class_name}/c{num_ref_points}/gps```.

## Infer query points 

To infer the query test point you can run:

```
python infer.py --class_name your_class_name
```

A full list of the subcomands can be found by running ```python infer.py --help```.

# Evaluation

To evaluate the infered point clouds against the ground truth and compute the metrics you can run:

```
python eval.py --class_name class_name
```
Again a full list of the subcomands can be found by running ```python eval.py --help```. By running this script the best models (i.e. the results for the optimal number of reference points with respect to best metric's error) will be exctracted as .ply files in ```./data/models/est_models/{class_name}```.
The total metrics will be saved under ```./data/results/{class_name}/total_scores.csv``` and they will also be printed in the terminall.
```
        Chairs Statistics          
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric    ┃   Mean    ┃  Median   ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cd        │ 0.000194  │ 0.000191  │
│ emd       │ 0.017624  │ 0.014774  │
│ precision │ 83.789200 │ 83.322800 │
│ recall    │ 91.646533 │ 91.149200 │
│ f1_score  │ 87.529484 │ 88.061813 │
└───────────┴───────────┴───────────┘
```
# Point cloud visualization

To visualize the output point clouds you can run:

```
python vis.py --num-points num_points # Subset of points to visualize \
```
We utilize Blender's API to render the point clouds representing the points as spheres.

## Score 6D poses




