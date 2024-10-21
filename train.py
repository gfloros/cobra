from rich.console import Console
import rich.progress
import tyro
from utils.model_3d import *
import common
from os.path import join as jn
from utils.io import load_config, make_dir, Train_args, log_event, log_debug
import gpytorch
from model import ExactGPModel
import torch
import rich
from utils.metrics import Metrics
import dataclasses
from torch.utils.data import DataLoader, TensorDataset

# gpytorch.settings.cholesky_jitter(float_value=1e-3)
@dataclasses.dataclass
class Train_args:

    # Class name to train
    class_name: str
    # initial learning rate
    init_lr: float = 0.1
    # CG iterations
    cg_iters: int = 1000
    # reference point selection method
    reference_point_selection_method: str = "kmeans"
    # Cluster overlap percentage
    cluster_overlap: float = 0.2
    # NUmber of steps to train
    num_steps: int = 250
    # Minimum number of reference points
    min_num_classes: int = 8
    # Maximum number of reference points
    max_num_classes: int = 14
    # Step to increase the number of reference points to try
    step: int = 2
    # Log loss
    log_loss: bool = False


def train_GPs_per_ref_point(
    class_name: str,
    train_model: str,
    cg_iters: int = 2000,
    reference_point_selection_method: str = "kmeans",
    num_classes: int = 10,
    cluster_overlap: float = 0.2,
    num_iters: int = 250,
    init_lr: float = 0.1,
    log_loss: bool = False,
):

    # load 3D points
    log_event("Loading 3D points...")
    points = load_point_cloud(
        jn(common.MODELS_PATH, "train", class_name, train_model) + ".ply"
    )

    # make result dir
    result_path_train_model = jn(
        common.RESULTS_PATH, class_name, train_model, "c" + str(num_classes)
    )
    make_dir(result_path_train_model)
    # get reference points
    log_event(
        f"Computing reference points using method {reference_point_selection_method}..."
    )
    if reference_point_selection_method == "kmeans":
        labels, centers, kmeans = findCentersKmeans(
            points.astype("double"), num_classes, savePath=result_path_train_model
        )
    elif reference_point_selection_method == "skeleton":
        skeleton_points_path = jn(common.MODELS_PATH, "skeletons", train_model + ".ply")
        labels, centers = load_skeleton_points(points, skeleton_points_path)
        print(centers)
    else:
        raise NotImplementedError(
            f"Reference point selection method {reference_point_selection_method} not implemented."
        )

    log_event("Create overlaping regions...")
    if cluster_overlap is not None:
        log_event("Computing overlapping regions...")
        points, labels = interClusterOverlap(
            points,
            labels,
            centers,
            obbtree=create_vtk_orbtree_from_mesh(
                jn(common.MODELS_PATH, "original", class_name, train_model + ".ply")
            ),
            overlap_radius_ratio=cluster_overlap,
        )

    log_event("Form the training input for the gaussian processes...")
    distances = distance_from_centers(points, centers, labels)
    _, phis_thetas_train, ds, _ = direction_distance_given_class(
        points, distances, centers, labels
    )

    trained_gps, likelihoods = [], []
    metrics = Metrics(["cd"])
    
    # train the gaussian processes
    with gpytorch.settings.max_cg_iterations(cg_iters):
        for cls in range(0, len(centers)):
            log_debug("Training GP model for class " + str(cls))

            # load points of class {cls} to tensor.
            phis_thets_cls_train_t = torch.tensor(phis_thetas_train[cls],dtype=torch.float32, device='cuda')
            ds_train = torch.tensor(ds[cls],dtype=torch.float32, device="cuda")

            # set the likelihood and instantiate the model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(phis_thets_cls_train_t, ds_train, likelihood).cuda()

            # set the likelihood and the model to train mode
            model.train()
            likelihood.train()

            # set the loss fucntion to optimize the model
            emll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # set the optimizer and scheduler
            # optimizer = torch.optim.LBFGS(model.parameters(),max_iter=2000,lr=init_lr, line_search_fn="strong_wolfe")
            #optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
            optimizer = torch.optim.Adam([
                {'params': model.covar_module.parameters(), 'lr': init_lr},       # Learning rate for kernel params
                {'params': model.likelihood.parameters(), 'lr': init_lr}  # Learning rate for noise (likelihood)
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.1, patience=10, min_lr=1e-6
            )

            def closure():

                optimizer.zero_grad()
                output = model(phis_thets_cls_train_t)
                loss = -emll(output, ds_train)
                loss.backward()

                if log_loss:
                    log_debug(
                        "Iter %d/%d - Loss: %.3f  lengthscale: %.3f  noise: %.3f"
                        % (
                            i + 1,
                            num_iters,
                            loss.item(),
                            model.covar_module.base_kernel.lengthscale.item(),
                            model.likelihood.noise.item(),
                        )
                    )
                return loss

            # training loop
            for i in range(num_iters):
                optimizer.step(closure)
                scheduler.step(closure())

            trained_gps.append(model)
            likelihoods.append(likelihood)

            log_debug(f"[bold][green] Done training for GP model {cls}")

    make_dir(
        jn(
            common.RESULTS_PATH,
            class_name,
            train_model,
            "c" + str(centers.shape[0]),
            "gps",
        )
    )
    for idx, model in enumerate(trained_gps):
        checkpoint = {
            "model": model.state_dict(),
            "train_x": phis_thetas_train[idx],
            "train_y": ds[idx],
        }
        torch.save(
            checkpoint, jn(result_path_train_model, "gps", f"gp_model_{idx}.pth")
        )

if __name__ == "__main__":

    args = tyro.cli(Train_args)
    with tqdm.tqdm(
        total=len(os.listdir(common.MODELS_PATH + "/train/" + args.class_name)),
        desc=f"Training class {args.class_name}",
    ) as pbar:
        for model in os.listdir(common.MODELS_PATH + "/train/" + args.class_name):
            print(f"Training model {model}")
            for cls in range(args.min_num_classes, args.max_num_classes, args.step):
                train_GPs_per_ref_point(
                    args.class_name,
                    model.split(".")[0],
                    init_lr=args.init_lr,
                    reference_point_selection_method=args.reference_point_selection_method,
                    cluster_overlap=args.cluster_overlap,
                    num_classes=cls,
                    num_iters=args.num_steps,
                    log_loss=args.log_loss,
                )
            pbar.update(1)
    log_event("Training complete")