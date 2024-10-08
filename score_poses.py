import numpy as np
from COBRA import COBRA
import common
import tyro
import dataclasses
import pickle
from os.path import join as jn
import json
import glob
import os
from utils.vis_utils import *
from utils.io import loadEstimatorResults
from pose_vis.Renderer import Renderer
from pose_vis.utils import load_model

@dataclasses.dataclass
class score_args:

    # Path to Kmeans weights
    kmeans: str

    # class name
    class_name: str

    # Model name under common.RESULTS_PATH/class_name
    model: str

    # delta value
    delta: float = 0.02

    # Path to estimator files
    estimator_results: str


def run(args):

    # Prepare and load data
    # load camera intrinsics
    K = np.loadtxt(args.estimator_results + "/K.txt")

    # load the estimated poses
    with open(args.estimator_results + "/poses.json") as f:
        poses = json.load(f)

    # load the 2D-3D correspondences and associated images (for visualization)
    corrs = sorted(glob.glob(jn(args.estimator_results,'corr')+"/*corr"))
    images = glob.glob(jn(args.estimator_results,args.model,'images') + "/*.png")
    
    # load the COBRA model
    cobra = COBRA(
        jn(common.RESULTS_PATH, args.class_name, args.model),
        jn(
            common.MODELS_PATH,
            "test",
            args.class_name,
            args.model.split("/")[0] + ".ply",
        ),
        delta=args.delta,
    )


    # initialize renderer
    renderer = Renderer(bufferSize=(640,480))
    renderer.load_shaders("./vis/shaders/basic_lighting_vrt.txt",
                        "./vis/shaders/basic_lighting.txt",
                        None)
    vertices, indices = load_model(jn(common.MODELS_PATH,'original',args.class_name,args.model+'.ply'))
    renderer.create_data_buffers(vertices,indices,attrs=[2,3,4])
    renderer.CreateFramebuffer(GL_RGB32F,GL_RGBA,GL_FLOAT)

    for img in images:

        renderer.glConfig()
        corr = jn(args.estimator_results,args.model,'corrs',str(int(os.path.basename(img).split(".")[0]))+"_corr.txt")
        
        # load the corr file into a pandas dataframe
        corr_df = loadEstimatorResults(corr)

        # get the corresponding pose
        pose_id = str(int(os.path.basename(corr).split('_')[0]))
        RT = np.eye(4)
        R = np.array(poses[pose_id][0]['cam_R_m2c']).reshape(3,3)
        T = np.array(poses[pose_id][0]['cam_t_m2c'])
        RT[:3,:3] = R
        RT[:3,-1] = T

        # get only the inliers
        #inliers = corr_df[corr_df.iloc[:,1].to_numpy().astype(int) ==1]
        
        # extract the 2D and 3D points
        points_3D = np.column_stack((corr_df['X'],corr_df['Y'],corr_df['Z']))
        points_2D = np.column_stack((corr_df['x'],corr_df['y']))
        if 'conf' in corr_df.columns:
            estimator_weights = corr_df['conf']
        else:
            estimator_weights = None

        likelihoods,distances = cobra.score_pose(points_2D,
                                                 points_3D,
                                                 RT[:3,:3],
                                                 K,
                                                 estimator_weights)
        
        # visualize and save results
        renderPose(vertices.reshape(-1,3),
                indices,
                renderer,
                objID=args.model,
                conf=likelihoods.mean(),
                threshold=cobra.conf_lower_bound,
                resolution=(640,480),
                RT= RT,
                K = K.reshape(3,3),
                savePath= jn(args.estimator_results,args.model,'vis',pose_id) + ".png",
                mesh_color=[1.0, 0.5, 0.31],
                rgb_image=img
                )
        
if __name__ == "__main__":
    args = tyro.cli(score_args)
    run(args)
