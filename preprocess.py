from utils.model_3d import (
    sample_points_from_mesh,
    sample_surface_points_trimesh,
    sample_visible_surface,
    load_mesh,
    fitModel2UnitSphere,
    get_model_size,
)
from utils.io import make_dir, log_event
import tyro
import dataclasses
from typing import List
import os
import common
from os.path import join as jn
from rich.console import Console
import open3d as o3d
import shutil

CONSOLE = Console()


@dataclasses.dataclass
class sample_points_args:

    # class name
    class_name: str

    # path to original meshes folder
    data_path: str = common.MODELS_PATH + "/original"

    # normalize the point cloud
    normalize: bool = True

    # Number of sampling points for train and test splits
    num_samples: List[int] = dataclasses.field(default_factory=lambda: [10000, 250000])

    visible: bool = True


def run(args):

    make_dir(common.MODELS_PATH + "/train")
    make_dir(common.MODELS_PATH + "/test")

    make_dir(common.MODELS_PATH + "/train/" + f"/{args.class_name}")
    make_dir(common.MODELS_PATH + "/test/" + f"/{args.class_name}")
    make_dir(common.MODELS_PATH + "/normalized" + f"/{args.class_name}")

    for model in os.listdir(jn(args.data_path, args.class_name)):

        if args.normalize:
            # save the normalized model
            original_model = load_mesh(
                jn(common.MODELS_PATH, "original", args.class_name, model)
            )
            normalized_vetcies = fitModel2UnitSphere(original_model[0])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(normalized_vetcies)
            mesh.triangles = o3d.utility.Vector3iVector(original_model[1])

            # Optional: Compute normals for better visualization
            mesh.compute_vertex_normals()

            # Step 4: Save the mesh to a file (e.g., .ply format)
            o3d.io.write_triangle_mesh(
                common.MODELS_PATH
                + "/normalized"
                + f"/{args.class_name}/{os.path.basename(model)}",
                mesh,
            )
        else:

            # just coppy the original model into the normalized models path
            shutil.copy(
                jn(args.data_path, args.class_name, model),
                jn(
                    args.data_path.replace("original", "normalized"),
                    args.class_name,
                    model,
                ),
            )

        log_event(f"Sampling points for model {model}")
        # sample training points
        if args.visible:
            log_event("Sampling training points")
            sample_visible_surface(
                num_sample_centers=100,
                mesh_path=jn(
                    args.data_path.replace("original", "normalized"),
                    args.class_name,
                    model,
                ),
                num_train=args.num_samples[0],
                num_test=args.num_samples[1],
                model_class=args.class_name,
                scale_fn=1.0 if args.normalize else get_model_size,
                model_name=model,
                save_sampling_positions=True,
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
