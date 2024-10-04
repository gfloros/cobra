from utils.metrics import Metrics, compute_metrics_per_class
import tyro
import dataclasses
import common
import os
import pandas as pd
import shutil
import tqdm
from utils.io import make_dir
from utils.model_3d import load_point_cloud, fitModel2UnitSphere
from os.path import join as jn
import time
from utils.report import create_and_render_tables

@dataclasses.dataclass
class eval_args:

    # Metrics to compute
    metrics: list = dataclasses.field(default_factory=lambda: ['cd', 'emd', 'precision', 'recall', 'f1_score'])

    # Points to evaluate each metric
    points_per_metric : list = dataclasses.field(default_factory= lambda: [30000,500,-1,-1,-1])

    # Class name to evaluate
    class_name: str = "all"
def run(args):

    metrics = Metrics(args.metrics)
    if args.class_name == "all":
        classes = os.listdir(jn(common.RESULTS_PATH))
    else:
        classes = [args.class_name]
    for class_name in classes:
        make_dir(jn(common.MODELS_PATH,
                'est_models',
                class_name))
        for model in tqdm.tqdm(os.listdir(jn(common.RESULTS_PATH,class_name))):
            if os.path.isdir(jn(common.RESULTS_PATH,class_name,model)):
                
                pcd_gt = load_point_cloud(jn(common.MODELS_PATH,"test",class_name,model+'.ply'))
                pcd_gt = fitModel2UnitSphere(pcd_gt,buffer=1.03)
                
                min_cd_error = float('inf')
                best_rfp_metrics = None
                best_rfp = None
                for rfp in os.listdir(jn(common.RESULTS_PATH,class_name,model.split('.')[0])):
                    if os.path.isdir(jn(common.RESULTS_PATH,class_name,model.split('.')[0],rfp)):
                        # load the predicted point cloud for each number of reference points
                        pcd_est = load_point_cloud(jn(common.RESULTS_PATH,class_name,model.split('.')[0],rfp,'infer',"est_points.ply"))

                        # compute metrics for model
                        rfp_metrics = metrics.compute(pcd_est,
                                                    pcd_gt,
                                                    points_per_metric=args.points_per_metric)
                        if rfp_metrics.get("cd") < min_cd_error:
                            min_cd_error = rfp_metrics.get("cd")
                            best_rfp_metrics = rfp_metrics
                            best_rfp = rfp

                # save the best rfp metrics
                if best_rfp_metrics is not None:
                    data = {"best_num_ref_points":[best_rfp]}
                    data.update(best_rfp_metrics)
                    df = pd.DataFrame(data)
                    output_csv_path = jn(common.RESULTS_PATH,
                                        class_name,model,
                                        "best_metrics.csv")
                    df.to_csv(output_csv_path,index=False)

                    # extract the mesh that corresponds to the best CD error
                    best_num_cls = df["best_num_ref_points"][0]
                    shutil.copy(
                        jn(
                            common.RESULTS_PATH,
                            class_name,
                            model,
                            best_num_cls,
                            'infer',
                            "est_points.ply",
                        ),
                        jn(
                            common.MODELS_PATH,
                            'est_models',
                            class_name,
                            model.split(".")[0] + ".ply",
                        ),
                    )
        summary = compute_metrics_per_class(class_name)
        create_and_render_tables(summary,class_name)
        summary.to_csv(jn(common.RESULTS_PATH, class_name, "total_scores.csv"))
        #print(summary)

if __name__ == "__main__":
    args = tyro.cli(eval_args)
    run(args)