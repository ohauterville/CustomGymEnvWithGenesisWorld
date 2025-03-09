"""
This code is to delete the logs and models folder of previous runs.
Run:
python delete.py --run_name $RUNNAME

Usually run_name looks like 02281200_PPO (date of the run in MMDDHHMM + _ai_model)
"""

import os
import shutil
import argparse


def delete_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="run name")

    # Parse the arguments
    args = parser.parse_args()

    models_folder = "models/" + args.run_name
    logs_folder = "logs/" + args.run_name

    try:
        delete_folder(models_folder)
    except Exception as e:
        print(e)

    try:
        delete_folder(logs_folder)
    except Exception as e:
        print(e)
