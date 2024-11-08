import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import trimesh
import vtk
import tqdm
import common
from typing import Tuple, Callable, Union
from numpy.typing import NDArray
from os.path import join as jn
from utils.io import log_event


def ray_intersects_mesh_twice_vtk(p, c, overlapping_radius, obbtree):
    """
    Check if a ray originating from point 'p' in direction 'c' intersects a mesh associated with 'obbtree' exactly twice.

    Parameters:
    p: list or tuple of length 3 - The origin of the ray (e.g., [x, y, z]).
    c: list or tuple of length 3 - The direction vector of the ray (e.g., [dx, dy, dz]).
    overlapping_radius: float - The radius tolerance for overlaps (not directly used by vtkOBBTree, but can be incorporated if needed).
    obbtree: vtkOBBTree - The oriented bounding box tree that provides spatial partitioning for the mesh.

    Returns:
    bool - True if the ray intersects the mesh exactly twice (entry and exit), False otherwise.
    """

    # Ensure 'p' and 'c' are valid 3D vectors
    if len(p) != 3 or len(c) != 3:
        raise ValueError(
            "The point 'p' and direction 'c' must both be 3-dimensional vectors."
        )

    # Create the start and end points for the ray in the direction of c
    ray_start = vtk.vtkPoints()
    p = np.array(p) - overlapping_radius * np.array(
        c
    )  # Move the start point back by a small amount to avoid self-intersections
    ray_start.InsertNextPoint(p)

    # Extend the ray in the direction 'c'. We multiply 'c' by a large scalar to ensure it reaches far enough.
    ray_end_point = c
    ray_end = vtk.vtkPoints()
    ray_end.InsertNextPoint(ray_end_point)

    # Use the OBBTree to compute the intersections
    points_intersection = vtk.vtkPoints()
    code = obbtree.IntersectWithLine(p, ray_end_point, points_intersection, None)

    # Check how many intersection points were found
    num_intersections = points_intersection.GetNumberOfPoints()

    # We are interested in exactly two intersections (entry and exit)
    return num_intersections > 1

def get_model_size(mesh_path: str) -> float:
    """Compute the diameter of the model.

    Args:
        mesh_path (str): Path to the mesh file.

    Returns:
        float: Diameter of the model.
    """
    
    # load the model and compute the min,max bounds
    model = o3d.io.read_triangle_mesh(mesh_path)

    minbb = np.asarray(model.vertices).min(axis=0)
    maxbb = np.asarray(model.vertices).max(axis=0)
    log_event(f"minbb = {minbb}, maxbb = {maxbb}")

    diameter = np.linalg.norm(maxbb - minbb)
    log_event(f"diameter = {diameter}")

    return diameter

def fibonacci_sphere(samples: int=1000, scale: float=1.0, savePath: str=None) ->NDArray:
    """ Generate points on a sphere using the Fibonacci spiral method.

    Args:
        samples (int, optional): Number of sampling reference points. Defaults to 1000.
        scale (float, optional): Take into account the scale of the object. Defaults to 1.0.
        savePath (str, optional): Path to save the sampling points for inspection. Defaults to None.

    Returns:
        NDArray: Sampling points on the sphere.
    """
    
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius * scale
        y = y * scale
        z = np.sin(theta) * radius * scale

        points.append([x, y, z])

    if savePath:
        write_ply(np.array(points), savePath)

    return np.array(points)


def sample_points_with_seed(points, num_samples, seed):
    np.random.seed(seed)
    indices = np.random.choice(len(points), size=num_samples, replace=False)
    sampled_points = points[indices]
    return sampled_points

def sample_visible_surface(
    num_sample_centers: int,
    mesh_path: str,
    num_train: int,
    num_test: int,
    model_class: str,
    scale_fn: Union[float, Callable],
    model_name: str,
    save_sampling_positions: bool = False,
) -> None:
    """ Sample points from the visible surface of the model. Create virtual camera position
    around the model and cast rays to sample points on the visible surface. 

    Args:
        num_sample_centers (int): Number of camera positions to sample.
        mesh_path (str): Path to the mesh file.
        num_train (int): Number of training points to sample.
        num_test (int): Number of testing points to sample.
        model_class (str): Model class name.
        scale_fn (Union[float, Callable]): Scale function to compute the model size. If the model
        is normalized in the unit sphere pass a scale_fn = 1. Otherwise, pass the get_model_size function.
        model_name (str): Name of the model in the class_name folder.
        save_sampling_positions (bool, optional): Save virtual camera positions. Defaults to False.
    
    Returns:
        None
    """
    # Load triangle mesh
    log_event("Loading triangle mesh")
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    # Create raycasting scene
    log_event("Creating raycasting scene")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # Generate camera positions on a sphere
    log_event("Generating camera positions on a sphere")
    if not callable(scale_fn):
        scale = 1.3
    else:
        scale = scale_fn(mesh_path)
    camera_positions = fibonacci_sphere(samples=num_sample_centers,
                                        scale = scale,
                                        savePath = jn(common.MODELS_PATH, "camera_positions.ply") if save_sampling_positions else None)

    # np.savetxt("camera_positions.xyz", camera_positions, delimiter=" ")
    # Parameters for the pinhole camera model
    fov_deg = 90
    width_px = 640
    height_px = 480
    center = [0, 0, 0]  # Look at the center of the model

    all_points = []

    log_event("Performing raycasting for each camera")
    for eye in tqdm.tqdm(
        camera_positions, total=camera_positions.shape[0], desc="Raycasting..."
    ):
        # Define the up vector
        up = [0, 1, 0]  # Assuming y-axis is up; adjust as necessary

        # Create rays using the pinhole camera model
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=fov_deg,
            center=center,
            eye=eye.tolist(),
            up=up,
            width_px=width_px,
            height_px=height_px,
        )
        # Perform raycasting
        result = scene.cast_rays(rays)

        # Extract intersection points
        hit = result["t_hit"].isfinite()
        points = rays[hit][:, :3] + rays[hit][:, 3:] * result["t_hit"][hit].reshape(
            (-1, 1)
        )
        all_points.append(points)

    log_event("Creating point cloud")
    pcd = o3d.geometry.PointCloud()
    all_ = o3d.core.concatenate(all_points, axis=0).numpy()
    points = np.array(all_)

    # Sample the first point cloud with a specific random seed
    log_event("Sampling the first point cloud with a specific random seed")
    sampled_points_1 = sample_points_with_seed(points, num_train, seed=42)
    log_event("Creating first point cloud")
    sampled_pcd_1 = o3d.geometry.PointCloud()
    log_event("Assigning points for first point cloud")
    sampled_pcd_1.points = o3d.utility.Vector3dVector(sampled_points_1)

    log_event("Sampling the second point cloud with a specific random seed")
    sampled_points_2 = sample_points_with_seed(points, num_test, seed=100)
    log_event("Creating second point cloud")
    sampled_pcd_2 = o3d.geometry.PointCloud()
    log_event("Assigning points for second point cloud")
    sampled_pcd_2.points = o3d.utility.Vector3dVector(sampled_points_2)

    # Save the sampled point cloud to a file (optional)
    log_event("Saving the sampled point cloud")
    o3d.io.write_point_cloud(
        common.MODELS_PATH
        + "/train/"
        + f"/{model_class}/"
        + model_name.split(".")[0]
        + ".ply",
        sampled_pcd_1,
    )
    o3d.io.write_point_cloud(
        common.MODELS_PATH
        + "/test/"
        + f"/{model_class}/"
        + model_name.split(".")[0]
        + ".ply",
        sampled_pcd_2,
    )


def create_vtk_orbtree_from_mesh(model: str):

    reader = vtk.vtkPLYReader()
    reader.SetFileName(model)
    reader.Update()
    poly_data = reader.GetOutput()

    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(poly_data)
    obb_tree.BuildLocator()

    return obb_tree


def sample_points_from_mesh(mesh_path, num_points, random_seed, output_path=None):

    np.random.seed(random_seed)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(num_points)

    if output_path is not None:
        o3d.io.write_point_cloud(output_path, pcd)

    return np.asarray(pcd.points)


def sample_surface_points_trimesh(mesh_path, num_samples, random_seed, output_path):

    np.random.seed(random_seed)
    mesh = trimesh.load_mesh(mesh_path, mesh_path.lower(), num_samples)

    points, _ = trimesh.sample.sample_surface_even(mesh, num_samples, seed=random_seed)

    pcd = trimesh.points.PointCloud(points)
    pcd.export(output_path)


def load_point_cloud(path: str):
    """Loads a point cloud from a file using open3d library.

    Args:
        path (str): Path to pointcloud path.

    Returns:
        NDArray: Point cloud points.
    """
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def load_mesh(path: str):
    """Loads a mesh from a file using open3d library.

    Args:
        path (str): Path to mesh file.

    Returns:
        tuple: Mesh vertices, triangles and open3d mesh object.
    """
    mesh = o3d.io.read_triangle_mesh(path)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def fitModel2UnitSphere(points, buffer=1.03):
    """Fits the model to a unit sphere.
    Points can be also padded using a buffer.

    Args:
        points (NDArray): Point cloud points.

    Returns:
        NDArray: Normalized point cloud points.
    """
    # find the centroid coordinates
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    zmin, zmax = points[:, 2].min(), points[:, 2].max()

    # center
    center = np.array([xmax + xmin, ymax + ymin, zmax + zmin]) / 2

    # trasform vertices
    points[:, 0] -= center[0]
    points[:, 1] -= center[1]
    points[:, 2] -= center[2]

    # calculate max distance
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances, axis=0)
    max_distance *= buffer

    points /= max_distance

    return points

def furthest_point_sampling(points, num_points):
    """Sample points from a point cloud using the furthest point sampling algorithm.

    Args:
        points (NDArray): Point cloud points.
        num_points (int): Number of points to sample.

    Returns:
        NDArray: Sampled points.
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
        points (NDArray): Points to classify.
        clusters (int): Number of reference points.
        savePath (str, optional): Path to save rcoordinates of reference points.
        Defaults to None.

    Returns:
        tuple(NDArray, NDArray, KMeans): Labels, centers and KMeans object.
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

                # if not ray_intersects_mesh_twice_vtk(p,c,overlaping_radius_per_class[idx_c],obbtree):
                # then include this point in this class
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
        points (NDArray): Point cloud points.
        centers (NDArray): Reference points.
        class_idxs (NDArray): Class indexes.

    Returns:
        NDArray: Distance of each point to the centers.
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
        points (NDArray): Point cloud points.
        distances (NDArray): Distance of each point to the centers.
        centers (NDArray): Coorfinates of reference points.
        cls_center_idxs (NDArray): Class indexes.
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


def cartesian_to_spherical(xyz):

    # Extract x, y, z coordinates
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Compute the radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute the polar angle (theta)
    theta = np.arccos(z / r)

    # Compute the azimuthal angle (phi)
    phi = np.arctan2(y, x)

    # Stack the results into a single array with shape (N, 3)
    spherical_coords = np.stack((r, theta, phi), axis=-1)

    return spherical_coords


def export_3D_points(points, filename):
    with open(filename, "w") as f:
        for i in points:
            f.write(f"{i[0]},{i[1]},{i[2]}\n")


def load_skeleton_points(points, file_path):
    centers = load_point_cloud(file_path)

    distances = np.linalg.norm(points[:, np.newaxis] - centers[np.newaxis, :], axis=-1)

    return np.argmin(distances, axis=1), centers


def write_ply(points, filename):
    """Write a point cloud to a PLY file.

    Args:
        points (NDArray): Point cloud points.
        filename (str): Output file path.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
