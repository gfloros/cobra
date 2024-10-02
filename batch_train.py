from train import train_GPs_per_ref_point
from infer import infer
import os
import common
import tyro
import dataclasses
from utils.io import make_dir
from utils.metrics import compute_metrics_per_class
from os.path import join as jn
import pandas as pd
import tqdm
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


@dataclasses.dataclass
class Batch_Train_args:

    # Class name to train
    class_name: str
    # Action to perform
    action: str = "all"
    # initial learning rate
    init_lr: float = 0.1
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
    # visualize point clouds
    vis: bool = False


def run(args):

    if args.action in ["train", "all"]:
        with tqdm.tqdm(
            total=len(os.listdir(common.MODELS_PATH + "/train/" + args.class_name)),
            desc=f"Training class {args.class_name}",
        ) as pbar:
            for model in os.listdir(common.MODELS_PATH + "/train/" + args.class_name):
                for cls in range(args.min_num_classes, args.max_num_classes, args.step):
                    train_GPs_per_ref_point(
                        args.class_name,
                        model.split(".")[0],
                        normalise=True,
                        init_lr=args.init_lr,
                        reference_point_selection_method="kmeans",
                        cluster_overlap=0.2,
                        num_classes=cls,
                        num_iters=args.num_steps,
                        log_loss=args.log_loss,
                    )
                pbar.update(1)
        print("Training complete")

    if args.action in ["infer", "all"]:
        ref_points_metrics = []
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
                                ref_point_selection_method="kmeans",
                                normalize=True,
                            )
                pbar.update(1)

    if args.vis:
        pass

if __name__ == "__main__":
    args = tyro.cli(Batch_Train_args)
    run(args)
