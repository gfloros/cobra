"""Module for 3D model processing and sampling."""
import pickle
import os
from pathlib import Path
from typing import Tuple

import fiblat
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
import tqdm
import vtk

import common
from utils.io import log_event


def get_model_size(mesh: o3d.t.geometry.TriangleMesh) -> float:
    """Compute the diameter of the model.

    Args:
        mesh (o3d.t.geometry.TriangleMesh): Mesh model.

    Returns:
        float: Diameter of the model.
    """
    minbb = np.asarray(mesh.vertices).min(axis=0)
    maxbb = np.asarray(mesh.vertices).max(axis=0)
    return np.linalg.norm(maxbb - minbb)


def perform_ray_casting(
    scene: o3d.t.geometry.RaycastingScene,
    camera_positions: np.ndarray,
) -> np.ndarray:
    """ Perform raycasting for each camera position.

    Args:
        scene (o3d.t.geometry.RaycastingScene): Raycasting scene.
        camera_positions (np.ndarray): Camera positions.

    Returns:
        List[np.ndarray]: List of points sampled from the visible surface.
    """
    all_points = []
    for eye in tqdm.tqdm(
        camera_positions,
        total=camera_positions.shape[0],
        desc="Raycasting..."
    ):
        # Create rays using the pinhole camera model
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=common.VCAMERA_FOV,
            center=common.SCENE_CENTER,
            eye=eye.tolist(),
            up=common.VCAMERA_UP_VECTOR,
            width_px=common.VCAMERA_WIDTH,
            height_px=common.VCAMERA_HEIGHT,
        )
        # Perform raycasting
        result = scene.cast_rays(rays)
        # Extract intersection points
        hit = result["t_hit"].isfinite()
        rays_start = rays[hit][:, :3]
        rays_dir = rays[hit][:, 3:]
        rays_hits = result["t_hit"][hit].reshape((-1, 1))
        points = rays_start + rays_dir * rays_hits
        all_points.append(points)
    return o3d.core.concatenate(all_points, axis=0).numpy()


def sample_points_with_seed(
    points: np.ndarray,
    num_samples: int,
    seed: int,
) -> o3d.geometry.PointCloud:
    """Sample points from a point cloud with a specific random seed.

    Args:
        points (np.ndarray): Point cloud points.
        num_samples (int): Number of points to sample.
        seed (int): Random seed.

    Returns:
        o3d.geometry.PointCloud: Sampled point cloud.
    """
    np.random.seed(seed)
    indices = np.random.choice(len(points), size=num_samples, replace=False)
    sampled_points = points[indices]
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    return sampled_pcd


def sample_visible_surface(
    mesh_path: str,
    train_output_path: str,
    test_output_path: str,
    save_camera_positions: bool = False,
) -> None:
    """ Sample points from the visible surface of the model.
        Create virtual camera position around the model and cast rays
        to sample points on the visible surface.

    Args:
        mesh_path (str): Path to the mesh file.
        train_output_path (str): Path to save the training point cloud.
        test_output_path (str): Path to save the test point cloud.
        save_camera_positions (bool, optional): Save virtual camera positions.
                                                Defaults to False.
    """
    # Load triangle mesh
    log_event("Loading triangle mesh")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Create raycasting scene
    log_event("Creating raycasting scene")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(np.asarray(mesh.vertices).astype(np.float32),
                        np.asarray(mesh.triangles).astype(np.uint32))

    # Generate camera positions on a sphere
    log_event("Generating camera positions on a sphere")
    scale = get_model_size(mesh)
    camera_positions = scale * fiblat.sphere_lattice(3, common.NUM_CENTERS)
    if save_camera_positions:
        output_path = Path(common.MODELS_PATH) / "camera_positions.ply"
        write_ply(camera_positions, output_path.absolute().as_posix())

    # Perform raycasting
    log_event("Performing raycasting for each camera")
    points = perform_ray_casting(scene, camera_positions)

    # Sample the first point cloud with a specific random seed
    log_event("Sampling the training/test point clouds")
    train_pcd = sample_points_with_seed(
        points,
        common.NUM_SAMPLES[0],
        seed=common.TRAIN_RANDOM_SEED
    )
    test_pcd = sample_points_with_seed(
        points,
        common.NUM_SAMPLES[1],
        seed=common.TEST_RANDOM_SEED
    )

    # Save the sampled point cloud to a file (optional)
    log_event("Saving the sampled point cloud")
    o3d.io.write_point_cloud(train_output_path, train_pcd)
    o3d.io.write_point_cloud(test_output_path, test_pcd)


def create_vtk_orbtree_from_mesh(model: str):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(model)
    reader.Update()
    poly_data = reader.GetOutput()
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(poly_data)
    obb_tree.BuildLocator()
    return obb_tree


def load_point_cloud(path: str):
    """Loads a point cloud from a file using open3d library.

    Args:
        path (str): Path to pointcloud path.

    Returns:
        np.ndarray: Point cloud points.
    """
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def normalize_mesh(model_path: str, output_path: str):
    """Normalize the model by fitting it to a unit sphere.

    Args:
        model_path (str): Path to the original model.
        output_path (str): Path to save the normalized model.
    """
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.translate(np.array([0.0, 0.0, 0.0]), relative=False)
    distances = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    scale = np.max(distances, axis=0) * common.SCALING_FACTOR
    mesh.scale(1 / scale, center=mesh.get_center())
    o3d.io.write_triangle_mesh(output_path, mesh)


def furthest_point_sampling(points, num_points):
    """Sample points from a point cloud using the furthest point sampling algorithm.

    Args:
        points (np.ndarray): Point cloud points.
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled points.
    """
    # initialize the list of indices
    sampled_idxs = [np.random.randint(len(points))]

    # loop over the number of points to sample
    for _ in range(1, num_points):
        # compute the distance between the sampled points and all other points
        distances = np.linalg.norm(
            points - points[sampled_idxs[-1]], axis=1
        )

        # find the index of the point with the largest minimum distance
        idx = np.argmax(np.min(distances, axis=0))

        # add the index to the list
        sampled_idxs.append(idx)

    return points[sampled_idxs]


def findCentersKmeans(points, clusters, init_method='k-means++', savePath=None):
    """Finds the centers of the clusters using KMeans.

    Args:
        points (np.ndarray): Points to classify.
        clusters (int): Number of reference points.
        savePath (str, optional): Path to save rcoordinates of reference points.
        Defaults to None.

    Returns:
        tuple(np.ndarray, np.ndarray, KMeans): Labels, centers and KMeans object.
    """
    if init_method == 'k-means++':
        kmeans = KMeans(n_clusters=clusters, init = init_method).fit(points)
    elif init_method == 'furthest_point':
        init_centers = furthest_point_sampling(points, clusters)
        kmeans = KMeans(n_clusters=clusters, init=init_centers).fit(points)
    if savePath is not None:
        with open(os.path.join(savePath, "kmeans.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        np.savetxt(
            os.path.join(savePath, "kmeans_centers.txt"),
            kmeans.cluster_centers_,
            delimiter=",",
        )

    return kmeans.labels_, kmeans.cluster_centers_, kmeans


def interClusterOverlap(
    points: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    obbtree,
    overlap_radius_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the inter-cluster overlap for a given set of points and cluster labels.

    Args:
        points (numpy.ndarray): The array of points.
        labels (numpy.ndarray): The array of cluster labels.
        centers (numpy.ndarray): The array of cluster centers.
        obbtree: The object representing the obbtree.
        overlap_radius_ratio (float, optional): The ratio of the overlap radius to the maximum point distance. Defaults to 0.05.

    Returns:
        numpy.ndarray: The array of classified points.
        numpy.ndarray: The array of modified labels.
    """

    overlaping_radius_per_class = []
    for idx, c in enumerate(centers):
        points_of_class = points[labels == idx]
        point_distances = np.linalg.norm(points_of_class - c, axis=-1)
        overlap_radius = (
            point_distances.max() + point_distances.max() * overlap_radius_ratio
        )
        overlaping_radius_per_class.append(overlap_radius)

    classified_points = []
    labels_mod = []
    excluded_points = 0
    for idx_c, c in enumerate(centers):
        points_per_class = []
        for idx, p in enumerate(points):
            # calculate the distance to each center
            distance = np.linalg.norm(p - c, axis=-1)
            if distance < overlaping_radius_per_class[idx_c]:
                points_per_class.append(p)
                labels_mod.append(idx_c)
            # else:
            # excluded_points += 1

        # finally append the array to the list
        classified_points.append(np.array(points_per_class))

    classified_points = np.concatenate(classified_points, axis=0)
    labels_mod = np.array(labels_mod)

    return classified_points, labels_mod


def distance_from_centers(points, centers, class_idxs):
    """Calculate the distance of each point to the centers.

    Args:
        points (np.ndarray): Point cloud points.
        centers (np.ndarray): Reference points.
        class_idxs (np.ndarray): Class indexes.

    Returns:
        np.ndarray: Distance of each point to the centers.
    """
    # reshape to use numpy broadcasting
    points = points[:, None, :]
    centers = centers[None, :, :]

    distances = np.linalg.norm(points - centers, axis=-1)

    return distances[np.arange(len(distances)), class_idxs]


def direction_distance_given_class(
    points,
    distances,
    centers,
    cls_center_idxs,
    saveClassPointsPath=None,
    return_scaled=False,
):
    """Calculate the direction and distance of each
    point to the centers given the class that the point is assigned.

    Args:
        points (np.ndarray): Point cloud points.
        distances (np.ndarray): Distance of each point to the centers.
        centers (np.ndarray): Coorfinates of reference points.
        cls_center_idxs (np.ndarray): Class indexes.
        saveClassPointsPath (bool, optional): Save points per class to file. Defaults to None.
        return_scaled (bool, optional): Normalize directions and disrtances. Defaults to False.

    Returns:
        tuple(List, List, List, List) : Return Clusters, phi_thetas, ds, cluster_indices lists per class.
    """

    clusters = []
    phi_thetas = []
    ds = []
    scalers = []
    cluster_indices = []
    unique_clusters = np.unique(cls_center_idxs)
    points = points - centers[cls_center_idxs]

    if saveClassPointsPath:
        f = open(os.path.join(saveClassPointsPath, "infer_classes.txt"), "w")
    for cluster in range(len(centers)):
        indices_for_cluster_i = np.where(cls_center_idxs == cluster)[0]
        cluster_points = points[indices_for_cluster_i]

        # print(f"Class {cluster}: {len(cluster_points)}")
        if len(cluster_points) == 0:
            phi_thetas.append([])
            ds.append([])
        else:
            if saveClassPointsPath:
                f.write(f"Class {cluster}: {len(cluster_points)}\n")
            clusters.append(cluster_points)

            phis = np.arctan2(cluster_points[:, 1], cluster_points[:, 0])  # polar angle
            thetas = np.arccos(
                cluster_points[:, 2] / (distances[indices_for_cluster_i])
            )  # azimuth

            if return_scaled:
                phis /= np.pi
                thetas /= 2 * np.pi
                distances[indices_for_cluster_i] /= distances.max()

            phi_thetas.append(np.column_stack((phis, thetas)))
            ds.append(distances[indices_for_cluster_i])
            cluster_indices.append(indices_for_cluster_i)

    if saveClassPointsPath:
        f.close()
    return clusters, phi_thetas, ds, np.concatenate(cluster_indices)


def xyz_from_direction_distance_class(phi_theta, ds, centers, class_idx):

    xyz = []
    x, y, z = spherical_coordinates_to_cartesian(phi_theta[:, 0], phi_theta[:, 1], ds)
    xyz = np.column_stack((x, y, z)) + centers[class_idx]

    return np.array(xyz)


def spherical_coordinates_to_cartesian(phi, theta, d):
    x = d * np.sin(theta) * np.cos(phi)
    y = d * np.sin(theta) * np.sin(phi)
    z = d * np.cos(theta)
    return x, y, z


def load_skeleton_points(points, file_path):
    centers = load_point_cloud(file_path)

    distances = np.linalg.norm(points[:, np.newaxis] - centers[np.newaxis, :], axis=-1)

    return np.argmin(distances, axis=1), centers


def write_ply(points, filename):
    """Write a point cloud to a PLY file.

    Args:
        points (np.ndarray): Point cloud points.
        filename (str): Output file path.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
