import os
import torch
from typing import  Literal
from typing import Tuple, Union, List
from torch._dynamo import OptimizedModule    

from TumorNetSolvers.reg_nnUNet.utilities.plans_handling.plans_handler import PlansManager
from TumorNetSolvers.preprocessing.target_handling import RegressionManager
from TumorNetSolvers.models.tumor_surrogate_net import TumorSurrogate
from TumorNetSolvers.models.ViT import CombinedVisionTransformer3D
from TumorNetSolvers.models.embeddings import PatchEmbed3D
from TumorNetSolvers.models.dynamic_Unets import get_network_from_plans_new


class InferenceManager:
    """
    Handles model initialization and checkpoint loading for inference.

    Attributes:
        device (torch.device): Device to run the model on (e.g., GPU or CPU).
        model (str): Specifies the model type to initialize (e.g., 'ViT', 'TumorSurrogate', 'nnUnet').
    """

    def __init__(self, plans: dict, configuration: str, model: Literal['ViT', 'TumorSurrogate', 'nnUnet'] = "ViT",
                 device: torch.device = torch.device(f'cuda:0'),  dataset_json :str ='', shape_data: torch.Size = None):
        self.device = device
        self.model = model

        # Setup plans and configurations
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.dataset_json= dataset_json
        self.shape_data = shape_data
        # Model initialization
        self.network = self._initialize_model()

        # Set the model to evaluation mode (important for inference)
        self.network.eval()

        # Optionally compile the model for optimization (torch.compile)
        if self._do_i_compile():
            print("Using torch.compile for speedup...")
            self.network = torch.compile(self.network)

        # Print status
        print(f"Initialized {self.model} model on device: {self.device}")

    def _do_i_compile(self):
        # Check device and platform conditions
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported MPS device")
            return False

        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        if os.name == 'nt':  # Windows
            if 'nnUNet_compile' in os.environ and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported.")
            return False

        # Check the environment variable to determine compilation status
        return os.environ.get('nnUNet_compile', 'true').lower() in ('true', '1', 't')
    
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int = 1,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        return get_network_from_plans_new(
            arch_class_name=architecture_class_name,
            arch_kwargs=arch_init_kwargs,
            arch_kwargs_req_import=arch_init_kwargs_req_import,
            input_channels=num_input_channels,
            output_channels=num_output_channels,
            inputs_shape=self.shape_data,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
    



    def _initialize_model(self):
        """Initializes the specified model."""
        if self.model == "ViT":
            return CombinedVisionTransformer3D(
                max_volume_size=self.configuration_manager.patch_size[0], patch_size=16, in_chans=1, num_classes=1000,
                embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, representation_size=None,
                distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D,
                norm_layer=None, act_layer=None, weight_init='', global_pool=False, param_dim=5
            ).to(self.device)
        elif self.model == "TumorSurrogate":
            model = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
            return model.to(self.device)
        elif self.model == "nnUnet":
            # Initialize nnUnet model for inference
            self.num_input_channels = RegressionManager.determine_num_input_channels(self.dataset_json)
            return self.build_network_architecture(
                "PlainConvUnetNew",
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model}")
        


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a pre-trained model from a checkpoint for inference.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Prepare state dict for loading
        state_dict = checkpoint['network_weights']
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v  # Handle DataParallel prefixes
            for k, v in state_dict.items()
        }
        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)
        # Load state dict into model
        print(f"Checkpoint loaded successfully from {checkpoint_path}.")
