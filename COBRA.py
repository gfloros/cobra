from model import ExactGPModel
import os
import glob
import pandas as pd
from os.path import join as jn
import pickle
import numpy as np
import math as mt
from numpy.typing import *
from sklearn.cluster import KMeans
from utils.transforms import *
from utils.model_3d import *
import torch
import gpytorch
from utils.model_3d import fitModel2UnitSphere

def compute_likelihood(point_3D_ref, p3d_observed, sigma=1.0, weight=1):

    eu_distance = np.linalg.norm(point_3D_ref - p3d_observed)
    likelihood = weight * mt.exp(-0.5 * (eu_distance**2) / (sigma**2))
    #likelihood = 1/(mt.sqrt(2*mt.pi  * sigma**2 )) * weight * mt.exp(-0.5 * (eu_distance**2) / (sigma**2))

    return eu_distance, likelihood

def compute_likelihood_exp(eu_distances, sigma):
    
    return mt.exp(-0.5 * (eu_distances.mean()**2) / (sigma**2))

def softmax(x):
    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class COBRA:
    def __init__(self, model_path, test_pcd_path) -> None:

        # load checkpoints with best results
        best_metrics = pd.read_csv(jn(model_path,'best_metrics.csv'))
        best_num_ref_points = best_metrics['best_num_ref_points'][0]

        self.models = []
        weights_path = jn(model_path,best_num_ref_points,'gps')
        for idx in range(0,int(best_num_ref_points.strip('c'))):
            self.models.append(ExactGPModel.load_from_file(jn(weights_path,f'gp_model_{str(idx)}.pth')).float().cuda())
            self.models[idx].eval()
        # load trained k-means
        kmeans_path = jn(model_path,best_num_ref_points,'kmeans.pkl')
        print(kmeans_path)
        with open(kmeans_path,'rb') as f:
            self.kmeans = pickle.load(f)
        self.centers = self.kmeans.cluster_centers_

        # computing sigma hat
        self.sigma_hat = self.compute_sigma_hat(jn(test_pcd_path,os.path.basename(model_path))+'.ply')

        self.conf_lower_bound = self.caclulate_confidence_lower_bound(delta=0.02)
        #print("CONF LOWER BOUND: ", self.conf_lower_bound)
    def compute_sigma_hat(self,gt_pcd):

        "Compute the STD of errors of the template from the GT point cloud."
        # load test point cloud
        gt_pcd = o3d.io.read_point_cloud(gt_pcd)
        gt_points = np.asarray(gt_pcd.points)

        gt_points = fitModel2UnitSphere(gt_points, buffer=1.03)
        # classify backprojected points to reference points
        class_idxs = self.kmeans.predict(gt_points.astype('double'))

        # phi,theta,d parameterization
        distances = distance_from_centers(gt_points,self.centers,class_idxs)
        _, phi_thetas, ds, sorted_indices = direction_distance_given_class(gt_points,
                                                       distances,
                                                       self.centers,
                                                       class_idxs)
        xyz_ = []
        for rfp in range(0, self.centers.shape[0]):
            # prepare torch tensors
            phis_thetas_test = torch.tensor(phi_thetas[rfp]).float().cuda()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if phis_thetas_test.shape[0] > 0:
                    observed_pred = self.models[rfp].likelihood(self.models[rfp](phis_thetas_test))
                    mean = observed_pred.mean

                    # Compute the 3D points of the infered points
                    xyz_predicted = xyz_from_direction_distance_class(phi_thetas[rfp],
                                                                    mean.cpu(),
                                                                    self.centers,
                                                                    rfp)
                    xyz_.append(xyz_predicted)

        xyz_ = np.concatenate(xyz_, axis=0)

        # compute the distances
        distances =  np.linalg.norm(xyz_ - gt_points[sorted_indices],axis=1)

        #print(distances.mean() , distances.std())
        return distances.std()


    def caclulate_confidence_lower_bound(self,
                                        delta,
                                        weights =None):
        # norm_factor = (1/ (delta**2)) * self.sigma_hat
        # wsum = self.sigma_hat * (1 - mt.exp(-((delta**2)/(2*self.sigma_hat))))

        # return norm_factor * wsum

        norm_factor = 1 / delta**2
        if weights is not None:
            wsum = np.sum(
                weights
                * self.sigma_hat**2
                * (1 - mt.exp(-((delta**2) / (2 * self.sigma_hat**2)))),
                axis=0,
            )
            return norm_factor * wsum
        else:
            return norm_factor * (self.sigma_hat**2) * (1 - mt.exp(-((delta**2) / (2 * self.sigma_hat**2))))




    def score_pose(self,
                   points2D: NDArray,
                   points3D: NDArray,
                   RT: NDArray,
                   K: NDArray,
                   weights: NDArray = None,
                   kmeans: KMeans = None,
                   delta: float = 5):

        back_proj_3D = []
        # back-project 2D points using the estimate pose
        # compute the 3D point across the ray with least squares
        for p2d,p3d in zip(points2D,points3D):
            back_proj_3D.append(transform_2D_to_3D(K,RT,np.append(p2d,1.0),p3d))

        back_proj_3D = np.asarray(back_proj_3D)



        # classify backprojected points to reference points
        class_idxs = self.kmeans.predict(back_proj_3D.astype('double'))

        # phi,theta,d parameterization
        distances = distance_from_centers(back_proj_3D,self.centers,class_idxs)
        _, phi_thetas, ds, sorted_indices = direction_distance_given_class(back_proj_3D,
                                                       distances,
                                                       self.centers,
                                                       class_idxs)
        xyz_ = []
        for rfp in range(0, self.centers.shape[0]):
            # prepare torch tensors
            phis_thetas_test = torch.tensor(phi_thetas[rfp]).float().cuda()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if phis_thetas_test.shape[0] > 0:
                    observed_pred = self.models[rfp].likelihood(self.models[rfp](phis_thetas_test))
                    mean = observed_pred.mean

                    # Compute the 3D points of the infered points
                    xyz_predicted = xyz_from_direction_distance_class(phi_thetas[rfp],
                                                                    mean.cpu(),
                                                                    self.centers,
                                                                    rfp)
                    xyz_.append(xyz_predicted)

        xyz_ = np.concatenate(xyz_, axis=0)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_)
        # o3d.visualization.draw_geometries([pcd])

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(back_proj_3D[sorted_indices])
        # o3d.visualization.draw_geometries([pcd])
        likelihoods = []
        distances = []
        raw_scores = []
        #sigmas_data_N = 0.03 # TODO: sigmas 3D model vs GP
        # compute the likelihood for each point
        # if weights are not provided default to 1/N

        if weights is None:
            weights = np.ones((len(xyz_)))
        
        
        for idx , (xyz_,xyz_ob) in enumerate(zip(xyz_,back_proj_3D[sorted_indices])):

            eu_distance, likelihood = compute_likelihood(
                xyz_, xyz_ob, sigma=self.sigma_hat, weight=weights[idx]
            )
            #print(f"Distance: {eu_distance}, Likelihood: {likelihood}")
            likelihoods.append(likelihood)
            distances.append(eu_distance)
        
        #print(np.array(likelihoods).mean(),np.array(distances).mean())

        return np.array(likelihoods) , np.array(distances)

