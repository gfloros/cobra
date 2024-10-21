import gpytorch
import torch
from rich.console import Console
import tyro
import common
from os.path import join as jn
import pickle
from utils.model_3d import *
from utils.io import make_dir, load_config, log_event, log_debug
from model import *
from utils.metrics import *
from utils.report import *
import dataclasses
from typing import List
import gc
import time

@dataclasses.dataclass
class Infer_args:

    # Class name to infer
    class_name: str
    # reference point selection method
    ref_point_selection_method: str = "kmeans"


def infer(
    class_name: str,
    test_model: str,
    ref_point_selection_method="kmeans",
    metrics: List = [
        "cd",
        "emd",
        "precision",
        "recall",
        "f1_score",
    ],
    points_per_metric: List = [30000, 500, -1, -1, -1],
):

    points = load_point_cloud(
        jn(common.MODELS_PATH, "test", class_name, test_model.split("/")[0] + ".ply")
    )

    if ref_point_selection_method == "kmeans":
        # load the trained k-means classifier weights for the specific model
        with open(
            jn(common.RESULTS_PATH, class_name, test_model, "kmeans.pkl"), "rb"
        ) as f:
            kmeans = pickle.load(f)
        ref_points = kmeans.cluster_centers_
        class_idxs = kmeans.predict(points.astype("double"))
    elif ref_point_selection_method == "skeleton":
        skeleton_points_path = jn(
            common.MODELS_PATH, "skeletons", test_model.split("/")[0] + ".ply"
        )
        class_idxs, ref_points = load_skeleton_points(points, skeleton_points_path)

    # get the distance from each point from their correspoding reference points
    distances = distance_from_centers(points, ref_points, class_idxs)
    _, phi_thetas, ds, sorted_indices = direction_distance_given_class(
        points, distances, ref_points, class_idxs
    )

    predicted_3d_points, total_predicted, total_gt = [], [], []
    mses, maes, cds, emds, css = [], [], [], [], []
    distances_diffs = []
    metrics_ = Metrics(metrics)

    # initialize metric dictionary
    metrics_per_class_dict = {}
    # load the trained gps
    for rfp in range(0, ref_points.shape[0]):
        log_debug(f"[bold][green] Making predictions for reference point {rfp}")

        gp_model_path = jn(
            common.RESULTS_PATH, class_name, test_model, "gps", f"gp_model_{rfp}.pth"
        )
        gp_model = ExactGPModel.load_from_file(gp_model_path)
        gp_model = gp_model.to("cuda")
        gp_model.eval()

        phi_thetas_test = torch.tensor(
            phi_thetas[rfp], dtype=torch.float32, device="cuda"
        )
        ds_test = torch.tensor(ds[rfp], dtype=torch.float32, device="cuda")

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if phi_thetas_test.shape[0] > 0:
                observed_pred = gp_model.likelihood(gp_model(phi_thetas_test))
                mean = observed_pred.mean
                lower, upper = observed_pred.confidence_region()

                # compute chamfer distance per point-cloud class
                xyz_predicted = xyz_from_direction_distance_class(
                    phi_thetas[rfp], mean.cpu(), ref_points, rfp
                )
                xyz_gt = points[class_idxs == rfp]
                total_gt.append(xyz_gt)
                total_predicted.append(xyz_predicted)

                # metrics_per_refp = metrics_.compute(
                #     xyz_gt, xyz_predicted, points_per_metric=points_per_metric, device="cuda"
                # )
                # metrics_per_class_dict[rfp + 1] = metrics_per_refp

            mean, lower, upper = mean.cpu(), lower.cpu(), upper.cpu()

            predicted_3d_points.append(
                xyz_from_direction_distance_class(
                    phi_thetas[rfp], mean, ref_points, rfp
                )
            )

    log_debug(["[bold][green] Done making predictions"])  
    make_dir(jn(common.RESULTS_PATH,class_name,test_model,'infer')) 
    write_ply(
        np.concatenate(predicted_3d_points, axis=0),
        jn(
            common.RESULTS_PATH, class_name, test_model, "infer", "est_points.ply"
        ),
    )
    log_debug("[bold][green] Exporting predicted 3D points...")

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":

    args = tyro.cli(Infer_args)
    make_dir(jn(common.RESULTS_PATH, args.class_name))
    make_dir(jn(common.MODELS_PATH,'est_models'))
    with tqdm.tqdm(
        total=len(os.listdir(jn(common.RESULTS_PATH, args.class_name))),
        desc=f"Inferring class {args.class_name}",
    ) as pbar:
        for model in os.listdir(jn(common.RESULTS_PATH, args.class_name)):

            # Initialize variables to track the minimum CD error and corresponding class
            min_cd_error = float("inf")
            best_total_metrics = None
            best_cls = None
            if os.path.isdir(jn(common.RESULTS_PATH, args.class_name, model)):
                for cls in os.listdir(
                    jn(common.RESULTS_PATH, args.class_name, model.split(".")[0])
                ):
                    if os.path.isdir(
                        jn(
                            common.RESULTS_PATH,
                            args.class_name,
                            model.split(".")[0],
                            cls,
                        )
                    ):
                        infer(
                            args.class_name,
                            model.split(".")[0] + "/" + cls,
                            ref_point_selection_method=args.ref_point_selection_method,
                        )
            pbar.update(1)
