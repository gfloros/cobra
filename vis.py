import bpy 
import numpy as np
import tyro
import os
import dataclasses
import common
from utils.io import stdout_redirected, make_dir
from utils.vis_utils import *
from utils.model_3d import fitModel2UnitSphere
from os.path import join as jn
import glob
import tqdm
import open3d as o3d

@dataclasses.dataclass
class vis_args:

    # Point cloud path
    point_clouds: str = jn(common.MODELS_PATH,'est_models') 
    # Number of points per model to visualize
    num_points: int = 10000
    # Ouput path
    out_path: str = jn(common.DATA_PATH,'vis')

def run(args):
    
    make_dir(args.out_path)
    clear_scene_objects(exceptions=[bpy.context.scene.camera.name])
    setup_gpu_rendering()
    # Position the camera
    move_camera_closer(7.5)

    # Create a pool of spheres
    sphere_pool = create_sphere_pool(args.num_points)

    for class_name in tqdm.tqdm(os.listdir(args.point_clouds), desc="Visualizing..."):
        make_dir(jn(args.out_path, class_name))
        for pcd in glob.glob(os.path.join(args.point_clouds,class_name,"*.ply")):
            # Load the point cloud and process it
            pcd_ = o3d.io.read_point_cloud(pcd)
            points = np.asarray(pcd_.points)

            # Fit the model to the unit sphere
            points = fitModel2UnitSphere(points, buffer=1.03)

            # Apply some rotations to the point cloud
            rotation = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            angle = np.radians(70)
            rotation2 = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            points = points @ rotation2 @ rotation

            np.random.shuffle(points)
            points = points[:args.num_points]

            # Place the spheres
            place_spheres(points,sphere_pool)

            # Render the scene
            output_image_path = jn(args.out_path, class_name, os.path.basename(pcd).split('.')[0] + ".png")

            bpy.context.scene.render.filepath = output_image_path
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
                "CUDA"  # or 'OPENCL' for AMD, or 'OPTIX' for RTX cards
            )
            bpy.context.scene.cycles.device = "GPU"

            # Select the GPU device for rendering
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for device in bpy.context.preferences.addons["cycles"].preferences.devices:
                device.use = (
                    True  # Enable the device (set to False if you want to disable CPU)
                )

            # Set GPU as the render device
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.scene.render.film_transparent = True
            scene = bpy.context.scene
            scene.render.resolution_x = 1200
            scene.render.resolution_y = 1200
            bpy.app.handlers.render_pre.clear()  # To avoid multiple prints of the same events
            # Disable all unnecessary logs
            bpy.app.debug_value = 0  # Sets debug level to 0, which minimizes verbosity
            scene.render.resolution_percentage = 100  # 100% of the specified resolution
            bpy.context.scene.cycles.samples = 128  # Adjust based on your need
            bpy.context.view_layer.update()
            bpy.context.scene.render.filepath = os.path.join(
                args.out_path, class_name,os.path.basename(pcd).replace(".ply", ".png")
            )
            bpy.ops.render.render(write_still=True)
            # Render the scene
            with stdout_redirected():
                bpy.ops.render.render(write_still=True)

if __name__ == "__main__":

    args = tyro.cli(vis_args)
    run(args)


