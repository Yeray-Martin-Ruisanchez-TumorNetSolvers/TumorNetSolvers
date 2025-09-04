# set_env.py
import os

def set_environment_variables():
    mount_dir = "/mnt/Drive3/jonas_zeineb/data_and_outputs"

    # Set environment variables
    os.environ['nnUNet_raw'] = os.path.join(mount_dir, "raw_data")
    os.environ['nnUNet_preprocessed'] = os.path.join(mount_dir, "preprocessed_data")
    os.environ['nnUNet_results'] = os.path.join(mount_dir, "results")

    print("Environment variables set successfully.")

if __name__ == "__main__":
    set_environment_variables()