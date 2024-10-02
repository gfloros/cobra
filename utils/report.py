from rich.console import Console
from rich.table import Table
import numpy as np
import os
import pandas as pd
from os.path import join as jn
import glob
import common



# TODO make a chart with the number of reference points 
# and the metrics. Mabe also calculate the mean and median

def generate_results_report(data_path):
    df_all_models = []
    for model in os.listdir(data_path):
        # Create an empty list to store the dataframes
        dfs = []
        for class_dir in sorted(os.listdir(jn(data_path, model))):
            if not os.path.isdir(jn(data_path, model, class_dir)):
                continue
            # Read each metrics_total.csv file to a dataframe
            df_total = pd.read_csv(jn(data_path, model, class_dir,'infer', 'metrics_total.csv'))
            df_total.index = [class_dir[1:]]
            # Add the dataframe to the list
            dfs.append(df_total)

        if dfs != []:
            # Concatenate the dataframes in the list into a single dataframe
            df_combined_model = pd.concat(dfs)
            df_combined_model.drop(columns=['Unnamed: 0'], inplace=True)
            df_all_models.append(df_combined_model)

            # Save the combined dataframe to a CSV file 
            print(df_combined_model)
            print(f"Saving the combined dataframe to {jn(data_path, model) + '/metrics_total_combined.csv'}")
            df_combined_model.to_csv(jn(data_path, model) + '/metrics_total_combined.csv', index=True)

    # Concatenate the dataframes in the list into a single dataframe
    df_all_models = pd.concat(df_all_models)
    df_all_models = df_all_models.groupby(df_all_models.index).mean()
    print(df_all_models)
    df_all_models.to_csv(jn(data_path) + '/metrics_summary.csv', index=True)


def create_and_render_tables(df,class_name):

        
    # Create a rich Table
    table = Table(title=f"{class_name} Statistics")

    # Add columns to the table
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Mean", justify="center", style="magenta")
    table.add_column("Median", justify="center", style="green")

    # Iterate over the DataFrame rows and add them to the table
    for metric, row in df.iterrows():
        table.add_row(metric, f"{row['mean']:.6f}", f"{row['median']:.6f}")
    
    # Render the table
    console = Console()
    console.print(table)

        
def get_deepSDF_metrics_for_selected_models(csv_file,train_models):
    
    # read csv files with metrics
    deepsdf_metrics = pd.read_csv(csv_file,delimiter=', ')
    
    
    # filter the dataframe using the model_names list
    deepsdf_metrics['shape'] = deepsdf_metrics['shape'].str.split('/').str[-1]
    print(deepsdf_metrics.columns)
    
    selected_cds = []
    count = 0
    for model in train_models:
       if model in deepsdf_metrics['shape'].values:
            # Get the corresponding CD value and add it to the list
            cd = deepsdf_metrics.loc[deepsdf_metrics['shape'] == model, 'chamfer_dist'].values[0]
            print(cd)
            selected_cds.append(cd)
            count += 1
    
    cds = np.array(selected_cds)
    print(f"Mean : {cds.mean()}, Median: {np.median(cds)}")
    print(count)
    
