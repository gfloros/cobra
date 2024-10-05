import os
import tyro
import bpy
import dataclasses
import numpy as np
import open3d as o3d
from utils.model_3d import fitModel2UnitSphere, findCentersKmeans
from utils.vis_utils import *
import tqdm
import common
from utils.io import stdout_redirected
import pickle
from os.path import join as jn

@dataclasses.dataclass
class render_args:

    # Path to input point cloud
    input_pcd: str
    # Path to output images
    out_images_path: str
    # Number of points to visualize
    num_points: int = 15000
    # Number of frames to render
    num_frames: int = 10


def run(args):
    clear_scene_objects(exceptions=[bpy.context.scene.camera.name])
    setup_gpu_rendering()
    # Create a pool of spheres
    
    #set_world_background()
    #add_empty_at_collection_center()

    pcd = o3d.io.read_point_cloud(args.input_pcd)
    points = np.asarray(pcd.points)
    light = bpy.data.lights.get("Light")
    light.energy = 1000  # Adjust intensity as needed
    light = bpy.data.objects.get("Light")
    #light.location = (0, 0, 10)
    camera = bpy.context.scene.camera
    camera.data.clip_start = 1  # Set near clipping plane
    camera.data.clip_end = 10000      # Set far clipping plane
    camera.location = (0,3, 0.3)
    camera.rotation_euler = (np.radians(90), 0, np.radians(180))
    points = rotate_points(points, angle_x=90,angle_y=0,angle_z=0)

    # Fit the model to the unit sphere
    points = fitModel2UnitSphere(points, buffer=1.03)
    np.random.shuffle(points)
    points = points[:args.num_points]

    with open(
            jn(common.RESULTS_PATH, "bunny","bunny", "c6", "kmeans.pkl"), "rb"
        ) as f:
            kmeans = pickle.load(f)
    ref_points = kmeans.cluster_centers_
    class_idxs = kmeans.predict(points.astype("double"))
    
    print(class_idxs)
    points = points + np.array([0, -0.5, 0.3])
    sphere_pool = create_sphere_pool(args.num_points, reference_points=ref_points,labels=class_idxs)

    place_spheres(points,sphere_pool)
    #add_white_cube(size=20, location=(0, 0, 9.3))
    # setup_circular_camera_path(radius=5,
    #                            location=(0, 0, 0),
    #                            init_camera_position=(0, 5, 3),
    #                            init_camera_rotation=(0, 0, 0),
    #                            path_duration=args.num_frames)
    
    # Set frame range for a smooth circular animation
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end =  args.num_frames  # 100 frames for one complete circle
    
    # Set output format to PNG (to render individual frames)
    scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = args.out_images_path+"/frame_"  # Output location for the frames
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
        "CUDA"  # or 'OPENCL' for AMD, or 'OPTIX' for RTX cards
    )
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    bpy.context.scene.cycles.device = "GPU"
    # Set GPU as the render device
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.render.film_transparent = True
    scene.render.resolution_percentage = 100  # 100% of the specified resolution
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.use_motion_blur = False

    # Render the animation (frames)
    #bpy.ops.render.render(animation=True)


    for i in tqdm.tqdm(range(args.num_frames), desc="Rendering..."):
        if i == 0:
            points = rotate_points(points, angle_x=0,angle_y=0,angle_z=0)
        else:
            points = rotate_points(points, angle_x=0,angle_y=0,angle_z=(360/args.num_frames), displacement=(0, -0.5, 0.3))
        place_spheres(points,sphere_pool)
        scene.render.filepath = args.out_images_path+"/frame_" + str(i)
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    args = tyro.cli(render_args)
    run(args)