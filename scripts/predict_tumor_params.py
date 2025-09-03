"""
This script performs inference and optimization on tumor parameter simulation using a specified deep learning model.
It applies continuous thresholding on tumor predictions, optimizes model parameters based on loss, and logs the results.

The main steps involved are:
1. Load the necessary settings and dataset based on the provided dataset name.
2. Perform inference using one of the supported models ('ViT', 'nnUnet', 'TumorSurrogate').
3. Optimize the model's parameters by comparing the predicted tumor regions with ground truth values and updating the model parameters through gradient descent.
4. Continuously threshold tumor predictions to generate binary masks.
5. Log metrics such as loss, gradients, and Dice score at each optimization step.
6. Save visualizations of the tumor predictions, ground truth, and their differences.
7. Save the input data, predictions (masked and thresholded), and ground truth as NIfTI files.
8. Log the results using Weights and Biases (wandb) for experiment tracking.

The script processes only the first batch of the dataset in each run.
Example usage:
    DATASET_NAME = "Dataset500_Brain"
    MODEL = "nnUnet"
    DATA_FOLDER = "/path/to/dataset"
    OUTPUT_BASE = "/path/to/output"
    device = torch.device('cuda:0')
    signature = "experiment_1"

    infer_parameters(DATASET_NAME, MODEL, DATA_FOLDER, OUTPUT_BASE, device, signature)
"""
#%%
import os
import time

from set_env import set_environment_variables
set_environment_variables()

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import wandb

from torch.utils.data import DataLoader
from TumorNetSolvers.inference.inference_manager import InferenceManager
from TumorNetSolvers.inference.inference_utils import CustomDataset, get_settings_and_file_paths
from TumorNetSolvers.utils.metrics import compute_dice_score

# Setting environment variables
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

def cont_function(x, low=0.25, high=0.675, steepness=100):
    """
    Applies a continuous sigmoid-based thresholding function to smooth the transition between predicted tumor regions.
    
    The function uses a sigmoid transformation to create a smooth, continuous transition between the 
    `low` and `high` thresholds, providing a more refined and less binary thresholding approach."""

    return ((high - low) * F.sigmoid((x - high) * steepness) + 
            F.sigmoid((x - low) * steepness) * low)


def save_nifti(tensor, path, name):
    """Saves a tensor as a NIfTI file."""
    os.makedirs(path, exist_ok=True)
    nib.save(nib.Nifti1Image(tensor.cpu().numpy(), np.eye(4)), os.path.join(path, f"{name}.nii.gz"))


def infer_parameters(dataset_name: str, model: str, data_folder: str, output_base: str, device: torch.device = torch.device('cuda:0'), signature: str = '10k', chkpt: str = ''):
    """
    Performs inference and optimization with comprehensive logging
    """
    runtimeStart = time.time()
    
    # Validate model
    valid_models = ['ViT', 'nnUnet', 'TumorSurrogate']
    if model not in valid_models:
        raise ValueError(f"Invalid model '{model}'. Valid models are: {valid_models}")

    # Load settings and file paths

    #TODO test_keys logic should be modified here 
    plan, dataset_json, test_keys, parameters = get_settings_and_file_paths(dataset_name)

    # Set up the dataset and data loader
    dataset = CustomDataset(data_folder, test_keys)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(output_base, exist_ok=True)
    # Obtain the shape of the input
    shape_data = dataset[0][0].shape # First [0] gets the sample tuple, second [0] extracts the image (data['data']) from it

    # Create output directories
    output_folder = os.path.join(output_base, f"_{model}_{signature}")
    os.makedirs(os.path.join(output_folder, "optimizeOutputPatients"), exist_ok=True)

    # Initialize InferenceManager and load checkpoint
    infer_manager = InferenceManager(plan, configuration='3d_fullres', model=model, device=device, dataset_json=dataset_json, shape_data=shape_data)
    
    if chkpt and os.path.exists(chkpt):
        checkpoint_path = chkpt
    else:
        checkpoint_path = os.path.join(nnUNet_results, dataset_name, 'Trainer__nnUNetPlans__3d_fullres', 'fold_train_val', f'_{signature}_{model}', f'checkpoint_{model}_best_ema_loss.pth')
    
    infer_manager.load_checkpoint(checkpoint_path)
    model_network = infer_manager.network._orig_mod.to(device).eval()

    # Wandb configuration
    configForWandb = {
        'model': model,
        'dataset': dataset_name,
        'signature': signature,
        'optimizationSteps': 200,
        'learningRate': 0.01
    }
    wandb.init(project="tumor-parameter-optimization", config=configForWandb)
    wandb.watch(model_network)

    # Inference and Optimization Loop
    for batch in data_loader:
        if len(batch) == 4:
            data, mask, target, keys = batch
        elif len(batch) == 3:
            data, target, keys = batch
            mask = None
        else:
            raise ValueError("Unexpected batch structure")

        filter= lambda x : (x>0)*x
        target = filter(target)
        data = data.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device)

        batch_params = [parameters[key] for key in keys]
        batch_params = torch.stack(batch_params).to(device)
        
        # Initialize optimization
        params = batch_params.clone()
        params.requires_grad = True
        optimizer = optim.Adam([params], lr=0.01)

        # Detailed logging dictionary
        saveDict = {
            "patientName": keys[0],
            "logDicts": [],
            "wandbRunID": None,
            "runtime": 0
        }

        for step in range(200):
            optimizer.zero_grad()
            output = filter(model_network(data, params)[0]) if isinstance(model_network(data, params),list) else filter(model_network(data, params)) #handle deep supervision

            # Masking
            if mask:
                output = output * mask.to(device)
            
            output_continuous = output.clone().detach().to(device)

            # Apply thresholding
            thresholded_output = cont_function(output, low=0.25, high=0.675).to(device)
            loss = F.mse_loss(thresholded_output, target.to(device))
            grad_parameters, = autograd.grad(loss, params, retain_graph=True)
            
            loss.backward()
            optimizer.step()

            # Logging
            logDict = {
                "_loss": loss.item(),
                "_grad_parameters_mean": grad_parameters.mean().item()
            }

            # Parameter tracking
            labels = ["x", "y", "z", "muD", "muRho"]
            parameter_diff = params - batch_params
            for j in range(5):
                logDict[f"grad_parameters_{labels[j]}"] = grad_parameters[0,j].item()
                logDict[f"parameters_{labels[j]}"] = params[0,j].item()
                logDict[f"parameters_difference_{labels[j]}"] = parameter_diff[0,j].item()
            logDict["_parameters_difference_mean"] = parameter_diff.abs().mean().item()

            # Dice score calculation
            thresholds = np.linspace(0.1, 0.9, 9).tolist() + [0.24, 0.665]
            for threshold in thresholds:
                dice = compute_dice_score(output_continuous, target.to(device), threshold=threshold)
                logDict[f"dice_{round(threshold, 2)}"] = dice.item()

            wandb.log(logDict)
            saveDict["logDicts"].append(logDict)

            # Visualization (every 20 steps)
            if step % 20 == 0 or step == 0:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                slice= target.shape[-1]//2
                # Input tissue
                axes[0].imshow(data.cpu().detach().numpy()[0,0,:,:,slice], cmap='gray')
                pred_plot = thresholded_output.cpu().detach().numpy()[0,0,:,:,slice]
                axes[0].imshow(pred_plot, vmin=0, vmax=1, cmap='Blues', alpha=pred_plot)
                axes[0].set_title(f'Step {step}: Prediction')

                # Ground truth
                axes[1].imshow(data.cpu().detach().numpy()[0,0,:,:,slice], cmap='gray')

                gt_plot = target.cpu().detach().numpy()[0,0,:,:,slice]
                axes[1].imshow(gt_plot, vmin=0, vmax=1, cmap='Greens', alpha=gt_plot)
                axes[1].set_title(f'Step {step}: Ground Truth')

                # Difference
                axes[2].imshow(data.cpu().detach().numpy()[0,0,:,:,slice], cmap='gray')
                diff_plot = np.abs(pred_plot - gt_plot)
                axes[2].imshow(diff_plot, vmin=0, vmax=1, cmap='Reds', alpha=diff_plot)
                axes[2].set_title(f'Step {step}: Difference')

                plt.tight_layout()
                save_path = os.path.join(output_folder, "optimizeOutputPatients", keys[0])
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"step{step}.pdf"))
                plt.close()

        # Save outputs
        case_n = keys[0]
        save_path = os.path.join(output_folder, "optimizeOutputPatients", case_n)
        os.makedirs(save_path, exist_ok=True)

        save_nifti(data[0, 0], save_path, 'input_tissue')
        save_nifti(output[0, 0].detach(), save_path, 'prediction_masked')
        save_nifti(target[0, 0], save_path, 'output_ground_truth')
        save_nifti(thresholded_output[0, 0].detach(), save_path, 'prediction_thresholded')

        # Finalize logging
        saveDict["wandbRunID"] = wandb.run.id
        saveDict["runtime"] = time.time() - runtimeStart
        torch.save(saveDict, os.path.join(save_path, "logDict.pth"))

        wandb.save(os.path.join(save_path, "logDict.pth"))
        wandb.finish()
        break 


if __name__=="__main__":
    DATASET_NAME = "Dataset500_Brain"
    MODEL = "nnUnet"  # Can also be 'ViT' or 'TumorSurrogate'
    DATA_FOLDER = os.path.join(nnUNet_preprocessed, DATASET_NAME,"nnUNetPlans_3d_fullres")
    OUTPUT_BASE = os.path.join(nnUNet_results, DATASET_NAME, 'infer_params')
    
   
    DEVICE = torch.device('cuda:0')
    SIGNATURE = "experiment"  # Or any other identifier for your experiment
    CHECKPOINT= "/mnt/Drive3/jonas_zeineb/data_and_outputs/results/Dataset700_Brain/Trainer__nnUNetPlans__3d_fullres/fold_train_val/_10k_2_nnUnet/checkpoint_nnUnet_best_ema_dice.pth"  #inputting checkpoint is optional, else it will look for checkpoint correspoding to same dataset and model (ie look for checkpoint for _nnUnet under results/dataset500_Brain folder)
    infer_parameters(DATASET_NAME, MODEL, DATA_FOLDER, OUTPUT_BASE, DEVICE, SIGNATURE, chkpt=CHECKPOINT)


# %%
