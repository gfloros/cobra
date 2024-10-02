from utils.model_3d import (
    sample_points_from_mesh,
    sample_surface_points_trimesh,
    sample_visible_surface,
)
from utils.io import make_dir, log_event
import tyro
import dataclasses
from typing import List
import os
import common
from os.path import join as jn
from rich.console import Console

CONSOLE = Console()


@dataclasses.dataclass
class sample_points_args:

    # class name
    class_name: str

    # path to original meshes folder
    data_path: str = common.MODELS_PATH + "/original"

    # Number of sampling points for train and test splits
    num_samples: List[int] = dataclasses.field(default_factory=lambda: [10000, 250000])

    visible: bool = True


def run(args):

    make_dir(common.MODELS_PATH + "/train")
    make_dir(common.MODELS_PATH + "/test")

    make_dir(common.MODELS_PATH + "/train/" + f"/{args.class_name}")
    make_dir(common.MODELS_PATH + "/test/" + f"/{args.class_name}")
    for model in os.listdir(jn(args.data_path, args.class_name)):
        log_event(f"Sampling points for model {model}")
        # sample training points
        if args.visible:
            sample_visible_surface(
                num_sample_centers=100,
                mesh_path=jn(args.data_path, args.class_name, model),
                num_train=args.num_samples[0],
                num_test=args.num_samples[1],
                model_class=args.class_name,
                model_name=model,
            )
        else:
            sample_points_from_mesh(
                jn(args.data_path, args.class_name, model),
                args.num_samples[0],
                random_seed=42,
                output_path=common.MODELS_PATH
                + "/train/"
                + f"/{args.class_name}/"
                + model.split(".")[0]
                + ".ply",
            )
            # sample test points
            sample_points_from_mesh(
                jn(args.data_path, args.class_name, model),
                args.num_samples[1],
                random_seed=102,
                output_path=common.MODELS_PATH
                + "/test/"
                + f"/{args.class_name}/"
                + model.split(".")[0]
                + ".ply",
            )

    log_event("Done sampling points.")


if __name__ == "__main__":
    args = tyro.cli(sample_points_args)
    run(args)
