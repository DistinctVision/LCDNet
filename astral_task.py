import sys
from pathlib import Path
import click
import clearml
import attrs
import tempfile

from datasets.astral_dataset_reader import AstralDatasetReader
from mldatatools.dataset import Dataset, FramesInfo, Frames
from astral_record_db import record_embeddings


@click.command()
@click.option(
    "--task-name",
    "-n",
    "task_name",
    default="Frame database",
    type=str,
)
@click.option(
    "--location",
    "-l",
    default="m11",
    type=str,
)
@click.option(
    "--project-name",
    "-p",
    "project_name",
    default="Processing/Annotate",
)
@click.option(
    "--dataset-project",
    default="Datastorage",
)
@click.argument("dataset_id", default="ca4d994772454114863723bc9a0f7ff7", required=True)
@click.argument("new_db_dataset_name", default="db_positions", required=True)
def main(
    dataset_id,
    new_db_dataset_name,
    task_name,
    project_name,
    dataset_project,
    location
):
    task = clearml.Task.init(project_name=project_name, task_name=task_name)

    clearml_dataset = clearml.Dataset.get(dataset_id=dataset_id)
    dataset_path = clearml_dataset.get_local_copy()

    # Run inference! ##############################################
    dataset_reader = AstralDatasetReader(Path(dataset_path), location, ['ld_cc'])
    ###############################################################

    tmp_dir = tempfile.TemporaryDirectory()
    db_dataset = Dataset.empty(tmp_dir.name, new_db_dataset_name)
    if db_dataset.frames is None:
        db_dataset.frames = FramesInfo(Frames())

    record_embeddings(dataset_reader, db_dataset, visualize=False)
    db_dataset.write()

    # Save new dataset to clearml
    new_clearml_dataset = clearml.Dataset.create(
        dataset_name=new_db_dataset_name,
        dataset_project=dataset_project,
        parent_datasets=[],
    )
    new_clearml_dataset.add_files(
        tmp_dir.name,
        dataset_path="/",
    )

    new_clearml_dataset.upload()
    # new_clearml_dataset.add_tags("LCDNet")
    new_clearml_dataset.finalize()

    task.upload_artifact("Link to dataset:", new_clearml_dataset._task.get_output_log_web_page())
    task.set_parameter("final_dataset_id", new_clearml_dataset._task.task_id)

    return 0


if __name__ == "__main__":
    print("Run with attrs",  attrs.__version__)
    sys.exit(main())
