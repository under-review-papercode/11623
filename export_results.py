import pandas as pd
import wandb
import click
from tqdm import tqdm
import os

pd.set_option('display.max_columns', 20)

@click.command()
@click.option('--path', type=str, default="aiis-chair/PubMedBLIP", help='Name of the project.', nargs=1)
@click.option('--name', type=str,  help='Name of the run.', nargs=1)
@click.option('--count', type=int,  help='Number of runs to consider. If set to -1, considers all runs.', nargs=1)
@click.option('--data', type=str,  help='Name of the datasets to use. Options: all|ovqa|rad|slake', nargs=1)
@click.option('--output', type=str,  help='Path to the directory for saving the final csv.', nargs=1)

def main(path, name, count, data, output):
    api = wandb.Api()
    runs = api.runs(
            path=path,
            filters={"display_name": name}
            )

    # TODO: excluded shape, MRI, CT, X-Ray, since not in rad! And CHEST, HEAD, presence, other, attribute since not in SLAKE! Also, position, size, color, quantity not in OVQA!
    ## The question is, do we need them really, when we cannot report it for all datasets and compare?
    # targets = ['organ', 'plane', 'modality', 'position', 'abnormality', 'Close', 'Open', 'Mean']
    targets = ['organ','position', 'abnormality', 'size', 'modality', 'quantity', 'Close', 'Open', 'Mean']

    # if data.lower() == 'all':
    #     datasets = ['rad', 'slake', 'ovqa']
    # else:
    #     datasets = [data.lower()]
    datasets = ['rad', 'slake']

    if count != -1 and count <= len(runs):
        runs = runs[:count]
   
    final_df = pd.DataFrame(columns=[f'A_{i}' for i in range(len(datasets)*len(targets)+1)], index=range(len(runs)))  ## +1 for run_id column

    for i, run in tqdm(enumerate(runs)):
        run_history = run.history(samples=run.lastHistoryStep)
        run_results = pd.Series(data={'run_id': i}, index=['run_id'])
        for dataset in datasets:
            prefix = f'vqa/{dataset}/'
            key = f'{prefix}best'
            best_step = run_history[key].idxmax()
            best_epoch = run_history.iloc[[best_step]]['epoch']
            target_columns = [prefix+target for target in targets]
            dataset_history = run_history[target_columns].fillna(0)
            dataset_results = dataset_history.max(axis=0)
            run_results = pd.concat([run_results, dataset_results])
        if i == 0:
            final_columns = run_results.index.tolist()
            final_df.columns = final_columns
        final_df.loc[i, :] = run_results.values

    avg_row = ['Average'] + [final_df[col].mean() for col in final_columns if col != 'run_id']
    std_row = ['Std'] + [final_df[col].std() for col in final_columns if col != 'run_id']

    final_df.loc[len(final_df.index)] = avg_row 
    final_df.loc[len(final_df.index)] = std_row

    output_path = os.path.join(output, f'{name}_{data.lower()}.csv')

    print(final_df.iloc[0])
    # final_df.to_csv(output_path, index=False)
    # print("done")


if __name__ == "__main__":
    main()
