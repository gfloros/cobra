""" Preprocess 3D models by sampling points from the models. """

import dataclasses
import os
from pathlib import Path
import shutil
import tyro

from rich.console import Console

import common

from utils.model_3d import (
    normalize_mesh,
    sample_visible_surface,
)
from utils.io import make_dir, log_event

CONSOLE = Console()


@dataclasses.dataclass
class SamplePointsArgs:
    """ Arguments for sampling points from 3D models """
    class_name: str
    normalize: bool = True


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
        output_path = self.model_normalized_dir / model
        normalize_mesh(
            model_filepath.absolute().as_posix(),
            output_path.absolute().as_posix(),
        )

    def sample_visible_surface(self, model: str):
        """ Sample points from the visible surface of the model.

        Args:
            model (str): Model name.
        """
        model_filepath = self.model_normalized_dir / model
        train_output_path = self.model_training_dir / f"{Path(model).stem}.ply"
        test_output_path = self.model_test_dir / f"{Path(model).stem}.ply"
        sample_visible_surface(
            model_filepath.absolute().as_posix(),
            train_output_path.absolute().as_posix(),
            test_output_path.absolute().as_posix(),
            save_camera_positions=True,
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
            log_event("Sampling training points from visible surface")
            self.sample_visible_surface(model)


if __name__ == "__main__":
    sample_points_args = tyro.cli(SamplePointsArgs)
    preprocess = Preprocess(sample_points_args)
    preprocess.run()
