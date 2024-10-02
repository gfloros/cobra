import numpy as np
from scipy.spatial import KDTree
import ot
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from functools import wraps
import common
import os
import torch
from os.path import join as jn

TAU = 0.01

def torch_chamfer_distance(p1, p2) -> float:
    p1 = p1.unsqueeze(0)  # Add batch dimension
    p2 = p2.unsqueeze(0)  # Add batch dimension

    p1_squared = torch.sum(p1**2, dim=-1, keepdim=True)
    p2_squared = torch.sum(p2**2, dim=-1, keepdim=True)
    p1_p2 = torch.matmul(p1, p2.transpose(-1, -2))

    dist = p1_squared + p2_squared.transpose(-1, -2) - 2 * p1_p2
    chamfer_dist = torch.mean(torch.min(dist, dim=-1)[0]) + torch.mean(
        torch.min(dist, dim=-2)[0]
    )

    return chamfer_dist.item()

class Metrics:
    
    def __init__(self, metrics):
        self.metrics_to_compute = metrics

    def compute(self, p1, p2,points_per_metric):
        self.results = {}
        for metric,num_eval_points in zip(self.metrics_to_compute,points_per_metric):
            # filter p1, p2 to the number of points specified
            if num_eval_points != -1:
                if p1.shape[0] > num_eval_points:
                    random_indices = np.random.choice(range(p1.shape[0]),num_eval_points,replace=False)
                    p1_f = p1[random_indices]
                    p2_f = p2[random_indices]
            else:
                p1_f = p1
                p2_f = p2
            func = getattr(self, metric, None)
            if func:
                self.results[metric] = func(p1_f, p2_f)
            else:
                raise ValueError(f"Metric '{metric}' not found.")
        #print(self.results)
        return self.results
    
    def clear(self):
        self.results = {}
    def to_df(self):
        self.metrics_df = pd.DataFrame(self.results, index=[0])
        return self.metrics_df

    def rmse(self, p1, p2):
        return np.sqrt(np.mean((p1 - p2) ** 2))
    
    def mae(self, p1, p2):
        return np.mean(np.abs(p1 - p2))
    
    def emd(self, p1, p2):

        M = ot.dist(p1, p2)
        # equally weighted
        a, b = np.ones((len(p1),)) / len(p1), np.ones((len(p2),)) / len(p2)
        return ot.emd2(a, b, M)

    def cs(self, p1, p2):
        cos_sim = cosine_similarity(p1, p2)
        return np.mean(np.diagonal(cos_sim))
    

    def cd(self, p1, p2):

        tree = KDTree(p1)
        dist_p1 = tree.query(p2)[0]
        tree = KDTree(p2)
        dist_p2 = tree.query(p1)[0]
        return np.mean(np.square(dist_p1)) + np.mean(np.square(dist_p2))
        #return (dist_p1.mean() + dist_p2.mean()) / len(p1)

    def mean_p2p_distance(self, p1, p2, norm=2):
        # calculate the L2 norm of each point
        distances = np.linalg.norm(p1 - p2, axis=-1)
        return distances.mean(), np.median(distances), distances.std()
    
    def precision(self,p1,p2,tau=TAU):
        
        # create KDTree
        tree_gt = KDTree(p1)
        
        # compute the distance of the reconstructed points to the ground truth
        dist_gen_to_gt = tree_gt.query(p2)[0]
        
        # get the precision
        smaller_than_tau_pr = np.where(dist_gen_to_gt < tau)
        self.precision_ = np.count_nonzero(smaller_than_tau_pr) / p1.shape[0] * 100

        return self.precision_
    
    def recall(self,p1,p2,tau=TAU):
        
        # create KDTrees
        tree_gen = KDTree(p2)

        # compute the distance of the gt to the reconstructed point cloud
        dist_gt_to_gen = tree_gen.query(p1)[0]

        # get the recall
        smaller_than_tau_re = np.where(dist_gt_to_gen < tau)
        self.recall_ = np.count_nonzero(smaller_than_tau_re) / p2.shape[0] * 100

        return self.recall_
    
    def f1_score(self,p1,p2):
        
        precision = self.precision(p1,p2)
        recall = self.recall(p1,p2)
        # get the F1 - score
        try:
            f1_score = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0

        return f1_score
    
def compute_metrics_per_class(class_name):

    metrics_list = []
    for model in os.listdir(jn(common.RESULTS_PATH,class_name)):
        if os.path.isdir(jn(common.RESULTS_PATH,class_name,model)):
            # get the total metrics for each model
            df = pd.read_csv(jn(common.RESULTS_PATH,class_name,model,'best_metrics.csv'))
            metrics_list.append(df.iloc[:,1:])

    all_metrics_df = pd.concat(metrics_list,ignore_index=True)

    
    # Compute mean and median
    mean_metrics = all_metrics_df.mean()
    median_metrics = all_metrics_df.median()

    # Create a summary dataframe
    summary_df = pd.DataFrame({
        'mean': mean_metrics,
        'median': median_metrics
    })

    return summary_df