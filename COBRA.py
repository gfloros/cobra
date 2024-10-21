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
from typing import Tuple


def compute_likelihood(
    point_3D_ref, p3d_observed, sigma=1.0, weight=1
) -> Tuple[float, float]:
    """Compute the likelihood of the observed 3D point given the
    the backrojected 2D input point. The likelihood is computed as a
    weighted gaussian distribution. If the weight is set to 1, the
    likelihood is computed as a normal gaussian distribution.

    Args:
        point_3D_ref (_type_): Predcited 3D point derived from the trained GP.
        p3d_observed (_type_): Backprojected 3D point.
        sigma (float, optional): Standart deviation of the tamplate vs
        the ground truth test points cloud. Defaults to 1.0.
        weight (int, optional): Posible estimator weight associated
        with the current precidiction of the 2D-3D correspondece. Defaults to 1.

    Returns:
        Tuple[float,float]: eu_distance (Euclidean distance between the predicted 3D point and the observed 3D point.)
                            likelihood (The computed likelihood of the observed 3D point.)
    """

    eu_distance = np.linalg.norm(point_3D_ref - p3d_observed)
    likelihood = weight * mt.exp(-0.5 * (eu_distance**2) / (sigma**2))

    return eu_distance, likelihood


class COBRA:
    def __init__(self, model_path: str, test_pcd_path: str, delta: float) -> None:
        """
        Initialize the COBRA model. Load the trained GPs and K-means classifiers.
        Methods:
            compute_sigma_hat: Compute the STD of errors of the template from the GT point cloud.
            caclulate_confidence_lower_bound: Compute the confidence lower bound.
            score_pose: Score input 6D poses.

        Args:
            model_path (str): Path to the trained model in common.RESULTS_PATH + class_name + model_name.
            test_pcd_path (str): Path to the GT test point cloud.
            delta (float): Delta value for the confidence lower bound.
        """

        # load checkpoints with best results
        best_metrics = pd.read_csv(jn(model_path, "best_metrics.csv"))
        best_num_ref_points = best_metrics["best_num_ref_points"][0]

        self.models = []
        weights_path = jn(model_path, best_num_ref_points, "gps")
        for idx in range(0, int(best_num_ref_points.strip("c"))):
            self.models.append(
                ExactGPModel.load_from_file(
                    jn(weights_path, f"gp_model_{str(idx)}.pth")
                )
                .float()
                .cuda()
            )
            self.models[idx].eval()
        # load trained k-means
        kmeans_path = jn(model_path, best_num_ref_points, "kmeans.pkl")
        print(kmeans_path)
        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)
        self.centers = self.kmeans.cluster_centers_

        # computing sigma hat
        self.sigma_hat = self.compute_sigma_hat(test_pcd_path)
        print("SIGMA HAT: ", self.sigma_hat)

        self.conf_lower_bound = self.caclulate_confidence_lower_bound(delta=delta)
        # print("CONF LOWER BOUND: ", self.conf_lower_bound)

    def compute_sigma_hat(self, gt_pcd: str) -> float:
        """Compute the STD of errors of the template from the GT point cloud.

        Args:
            gt_pcd (str): Path to the GT test point cloud.

        Returns:
            float: Computed STD of pair wise distance errors used as the variance
            for the GP model.
        """

        # load test point cloud
        gt_pcd = o3d.io.read_point_cloud(gt_pcd)
        gt_points = np.asarray(gt_pcd.points)

        # classify backprojected points to reference points
        class_idxs = self.kmeans.predict(gt_points.astype("double"))

        # phi,theta,d parameterization
        distances = distance_from_centers(gt_points, self.centers, class_idxs)
        _, phi_thetas, ds, sorted_indices = direction_distance_given_class(
            gt_points, distances, self.centers, class_idxs
        )
        xyz_ = []
        for rfp in range(0, self.centers.shape[0]):
            # prepare torch tensors
            phis_thetas_test = torch.tensor(phi_thetas[rfp]).float().cuda()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if phis_thetas_test.shape[0] > 0:
                    observed_pred = self.models[rfp].likelihood(
                        self.models[rfp](phis_thetas_test)
                    )
                    mean = observed_pred.mean

                    # Compute the 3D points of the infered points
                    xyz_predicted = xyz_from_direction_distance_class(
                        phi_thetas[rfp], mean.cpu(), self.centers, rfp
                    )
                    xyz_.append(xyz_predicted)

        xyz_ = np.concatenate(xyz_, axis=0)

        # compute the distances
        distances = np.linalg.norm(xyz_ - gt_points[sorted_indices], axis=1)

        return distances.std()

    def caclulate_confidence_lower_bound(
        self, delta: float, weights: NDArray = None
    ) -> float:
        """Compute the confidence lower bound.

        Args:
            delta (float): Delta value for the confidence lower bound.
            weights (NDArray, optional): Pose estimator weights if available. Defaults to None.

        Returns:
            float: Confidence lower bound.
        """
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
            return (
                norm_factor
                * (self.sigma_hat**2)
                * (1 - mt.exp(-((delta**2) / (2 * self.sigma_hat**2))))
            )

    def score_pose(
        self,
        points2D: NDArray,
        points3D: NDArray,
        RT: NDArray,
        K: NDArray,
        weights: NDArray = None,
    ) -> Tuple[NDArray, NDArray]:
        """Score input 6D poses. The method backprojects 2D points using the estimate pose
        and computes the 3D point across the ray with least squares. The method classifies
        backprojected points to reference points and computes the likelihood of the observed
        3D point given the the backrojected 2D input point.

        Args:
            points2D (NDArray): Input 2D points.
            points3D (NDArray): Corresponding 3D points.
            RT (NDArray): Estimated pose.
            K (NDArray): Camera intrinsics.
            weights (NDArray, optional): Estimator weights if available. Defaults to None.

        Returns:
            Tuple[NDArray, NDArray]: Likelihoods and distances for each point.
        """

        back_proj_3D = []
        # back-project 2D points using the estimate pose
        # compute the 3D point across the ray with least squares
        for p2d, p3d in zip(points2D, points3D):
            back_proj_3D.append(transform_2D_to_3D(K, RT, np.append(p2d, 1.0), p3d))

        back_proj_3D = np.asarray(back_proj_3D)

        # classify backprojected points to reference points
        class_idxs = self.kmeans.predict(back_proj_3D.astype("double"))

        # phi,theta,d parameterization
        distances = distance_from_centers(back_proj_3D, self.centers, class_idxs)
        _, phi_thetas, ds, sorted_indices = direction_distance_given_class(
            back_proj_3D, distances, self.centers, class_idxs
        )
        xyz_ = []
        for rfp in range(0, self.centers.shape[0]):
            # prepare torch tensors
            phis_thetas_test = torch.tensor(phi_thetas[rfp]).float().cuda()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if phis_thetas_test.shape[0] > 0:
                    observed_pred = self.models[rfp].likelihood(
                        self.models[rfp](phis_thetas_test)
                    )
                    mean = observed_pred.mean

                    # Compute the 3D points of the infered points
                    xyz_predicted = xyz_from_direction_distance_class(
                        phi_thetas[rfp], mean.cpu(), self.centers, rfp
                    )
                    xyz_.append(xyz_predicted)

        xyz_ = np.concatenate(xyz_, axis=0)

        likelihoods = []
        distances = []

        if weights is None:
            weights = np.ones((len(xyz_)))

        for idx, (xyz_, xyz_ob) in enumerate(zip(xyz_, back_proj_3D[sorted_indices])):

            eu_distance, likelihood = compute_likelihood(
                xyz_, xyz_ob, sigma=self.sigma_hat, weight=weights[idx]
            )
            likelihoods.append(likelihood)
            distances.append(eu_distance)

        return np.array(likelihoods), np.array(distances)
