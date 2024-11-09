""" Preprocess 3D models by sampling points from the models. """

import dataclasses
import os
from pathlib import Path
import shutil
from typing import List
import tyro

from rich.console import Console
import open3d as o3d

import common

from utils.model_3d import (
    sample_points_from_mesh,
    sample_visible_surface,
    load_mesh,
    fitModel2UnitSphere,
    get_model_size,
)
from utils.io import make_dir, log_event

CONSOLE = Console()
NUM_SAMPLES = [10000, 250000]
TRAIN_RANDOM_SEED = 42
TEST_RANDOM_SEED = 102


@dataclasses.dataclass
class SamplePointsArgs:
    """ Arguments for sampling points from 3D models """
    class_name: str
    normalize: bool = True
    num_samples: List[int] = dataclasses.field(
        default_factory=lambda: NUM_SAMPLES
    )
    visible: bool = True


class Preprocess:
    """ Preprocess 3D models by sampling points from the models. """
    def __init__(self, cmd_args):
        self.sample_points_args = cmd_args
        self.original_dir = Path(common.MODELS_PATH) / "original"
        self.normalized_dir = Path(common.MODELS_PATH) / "normalized"
        self.training_dir = Path(common.MODELS_PATH) / "train"
        self.test_dir = Path(common.MODELS_PATH) / "test"
        name = self.sample_points_args.class_name
        self.model_original_dir = self.original_dir / name
        self.model_training_dir = self.training_dir / name
        self.model_test_dir = self.test_dir / name
        self.model_normalized_dir = self.normalized_dir / name
        self.models = os.listdir(self.original_dir / name)
        make_dir(self.training_dir)
        make_dir(self.test_dir)
        make_dir(self.normalized_dir)
        make_dir(self.model_training_dir)
        make_dir(self.model_test_dir)
        make_dir(self.model_normalized_dir)

    def normalize_model(self, model: str):
        """ Normalize the model by fitting it to a unit sphere.

        Args:
            model_path (str): Path to the model.
            output_path (str): Path to save the normalized model.
        """
        model_filepath = self.model_original_dir / model
        vertices, triangles = load_mesh(model_filepath.absolute().as_posix())
        normalized_vetcies = fitModel2UnitSphere(vertices)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(normalized_vetcies)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        output_path = self.model_normalized_dir / model
        o3d.io.write_triangle_mesh(output_path.absolute().as_posix(), mesh)

    def sample_visible_surface(self, model: str):
        """ Sample points from the visible surface of the model.

        Args:
            model (str): Model name.
        """
        model_filepath = self.model_normalized_dir / model
        sample_visible_surface(
            num_sample_centers=100,
            mesh_path=model_filepath.absolute().as_posix(),
            num_train=self.sample_points_args.num_samples[0],
            num_test=self.sample_points_args.num_samples[1],
            model_class=self.sample_points_args.class_name,
            scale_fn=get_model_size,
            model_name=model,
            save_sampling_positions=True,
        )

    def sample_mesh(self, model: str):
        """ Sample points from the mesh.

        Args:
            model (str): Model name.
        """
        model_filepath = self.model_normalized_dir / model
        train_output_path = self.model_training_dir / f"{Path(model).stem}.ply"
        test_output_path = self.model_test_dir / f"{Path(model).stem}.ply"
        sample_points_from_mesh(
            model_filepath.absolute().as_posix(),
            self.sample_points_args.num_samples[0],
            random_seed=TRAIN_RANDOM_SEED,
            output_path=train_output_path.absolute().as_posix(),
        )
        sample_points_from_mesh(
            model_filepath.absolute().as_posix(),
            self.sample_points_args.num_samples[1],
            random_seed=TEST_RANDOM_SEED,
            output_path=test_output_path.absolute().as_posix(),
        )

    def run(self):
        """ Run the preprocessing pipeline. """
        for model in self.models:
            if self.sample_points_args.normalize:
                log_event(f"Normalizing model {model}")
                self.normalize_model(model)
            else:
                shutil.copy(
                    self.model_original_dir / model,
                    self.model_normalized_dir / model,
                )
            if self.sample_points_args.visible:
                log_event("Sampling training points from visible surface")
                self.sample_visible_surface(model)
            else:
                log_event("Sampling training points from mesh")
                self.sample_mesh(model)


if __name__ == "__main__":
    sample_points_args = tyro.cli(SamplePointsArgs)
    preprocess = Preprocess(sample_points_args)
    preprocess.run()
