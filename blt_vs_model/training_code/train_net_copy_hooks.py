
"""
Training Script for Vision Models (BLT-VS and Baselines)

This script implements the full experimental pipeline for training and evaluating
vision models (e.g., BLT-VS, ResNet, CORnet, vNet) on large-scale image datasets
such as ImageNet or EcoSet.

High-Level Functionality:
-------------------------
1. Parses command-line arguments to configure:
   - Network architecture (e.g., BLT-VS, ResNet)
   - Recurrence settings (timesteps, top-down, lateral, skip connections)
   - Optimization hyperparameters (learning rate, batch size, epochs)
   - Dataset and augmentation choices

2. Builds a structured hyperparameter dictionary (hyp) to ensure
   reproducibility and consistent experiment configuration.

3. Loads dataset loaders (train/val/test) with specified augmentations.

4. Instantiates the selected network architecture dynamically,
   allowing fair comparison across different models.

5. Sets up:
   - Loss function (CrossEntropy with optional label smoothing)
   - Optimizer (e.g., Adam)
   - Learning rate scheduler (warmup + adaptive decay)
   - Mixed precision training (AMP)
   - Gradient clipping (for training stability)

6. Executes the main training loop:
   - Forward pass
   - Loss computation (averaged across timesteps for recurrent models)
   - Backpropagation (including recurrent gradient flow)
   - Optimizer step
   - Validation evaluation
   - Learning rate scheduling
   - Logging and checkpoint saving

7. After training completion:
   - Saves final model weights
   - Evaluates performance on the test set
   - Stores all metrics for later analysis

Scientific Role:
----------------
This script defines the experimental training protocol for all models.
It ensures identical optimization conditions across architectures,
allowing meaningful comparison of recurrent (BLT-VS) and feedforward
models.

In summary:
-----------
This file does not define the architecture itself.
It defines how the architecture learns.
"""


##################
### Setting up and training a BLT network modelling the ventral stream
# 224px inputs <-> 5deg visual angle
##################

##################
### Collecting some hyperparameters that can be passed through cmd
##################

import argparse
import sys
from tqdm import tqdm
import matplotlib
from datetime import datetime
matplotlib.use("Agg")  # Important for HPC / no GUI
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Obtaining hyps')

parser.add_argument('--network', type=str, default='blt_vs') # blt_vs / rn50 / others...
parser.add_argument('--timesteps', type=int, default=6) # 6 is the minimum for no bio_unroll, 12 is the minimum for bio_unroll
parser.add_argument('--identifier', type=int, default=1)
parser.add_argument('--lateral_connections', type=int, default=1)
parser.add_argument('--topdown_connections', type=int, default=1)
parser.add_argument('--skip_connections', type=int, default=0)
parser.add_argument('--bio_unroll', type=int, default=1)
parser.add_argument('--readout_type', type=str, default='multi')
parser.add_argument(
    "--dataset_mode",
    type=int,
    default=0,
    help="0 = EcoSet, 1 = FakeData, 2 = CIFAR100"
)

parser.add_argument('--dataset', type=str, default='ecoset')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_val_test', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--start_from_epoch', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--max_steps', type=int, default=-1)
parser.add_argument('--bottlenecks', type=str, default='', help='comma list like "V1->V2:144,V2->V3:160"')
parser.add_argument('--grad_clipping', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_patience', type=int, default=2)
parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of mini-batches to accumulate before optimizer step')
parser.add_argument('--dataset_path', type=str, default='', help='Override dataset base path (e.g. node-local copy). Empty = use default /share/klab/datasets/')
parser.add_argument("--ecoset_debug_subset", action="store_true")
parser.add_argument("--ecoset_debug_size", type=int, default=500)
parser.add_argument('--name', type=str, default='', help='Optional custom name for the run. Overrides the auto-generated name.')
parser.add_argument('--use_mixup', type=float, default=0.0, help='MixUp alpha (0 to disable, only used for rn50)')
parser.add_argument('--use_cutmix', type=float, default=0.0, help='CutMix alpha (0 to disable, only used for rn50)')
parser.add_argument('--ra_reps', type=int, default=0, help='Repeated Augmentation repetitions (0 to disable, only used for rn50)')
parser.add_argument('--optimizer_type', type=str, default='adam', help='Optimizer: adam | sgd (sgd recommended for rn50)')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
parser.add_argument('--use_ema', type=int, default=0, help='Use Exponential Moving Average model (1=on, 0=off)')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay factor')
parser.add_argument('--lr_scheduler_type', type=str, default='linearfit', help='LR scheduler: linearfit | cosine')
parser.add_argument('--gradient_checkpointing', type=int, default=0, help='Enable gradient checkpointing to reduce memory (1=on, 0=off). Trades ~30%% compute for ~3-4x memory savings.')

args = parser.parse_args()

##################
### Importing required packages
##################

import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import time
import json
import inspect
from helpers.helper_funcs import get_Dataset_loaders, create_folders_logging, LinearFitScheduler, compute_first_signal
from models.helper_funcs import get_network_model, get_optimizer, eval_network, compute_accuracy, adaptive_gradient_clipping, calculate_flops
import gc

##################
### Listing all hyperparameters
##################

base_lr = args.learning_rate

if args.network == 'vNet': # vNet takes 128px images as inputs
    print('Working with 128px inputs')
    augmenter_train = {'resize_224','crop_224','resize_128','blurring','hflip','trivialaug','normalize'}
    augemnter_val_test = {'resize_224','centercrop_224','resize_128','normalize'}
elif args.network == 'rn50':
    print('Working with 176px train / 224px val inputs (ImageNet V2 recipe)')
    augmenter_train = {'randomresizedcrop_176','hflip','trivialaug','imagenet_normalize','random_erasing'}
    augemnter_val_test = {'resize_232','centercrop_224','imagenet_normalize'}
else: 
    # as imagenet images are not square, here we first rescale smaller axis to 224 and then crop.
    print('Working with 224px inputs')
    augmenter_train = {'resize_224','crop_224','blurring','hflip','trivialaug','normalize'}
    augemnter_val_test = {'resize_224','centercrop_224','normalize'}

hyp = {
    'dataset': {
        'name': args.dataset, # name of the dataset - ecoset/imagenet
        'dataset_path': args.dataset_path if args.dataset_path else '/share/klab/datasets/', # Folder where dataset exists (end with '/')
        'augment': augmenter_train, # Mention augmentations to be used here during training - blurring (always first), trivialaug, autoaugment, randaugment, normalize (always last)
        'augment_val_test': augemnter_val_test, # Mention augmentations to be used here during validation/testing
    },
    'network': {
        'name': args.network, # network to be used
        'identifier': str(args.identifier), # identifier in case we run multiple versions of the net
        'timesteps': args.timesteps, # number of timesteps to unroll the RNN
        'lateral_connections': args.lateral_connections, # whether to use lateral connections
        'topdown_connections': args.topdown_connections, # whether to use topdown connections
        'skip_connections': args.skip_connections, # whether to use skip connections
        'bio_unroll': args.bio_unroll, # whether to unroll the network in a biologically plausible manner
        'readout_type': args.readout_type # whether to use a single or multiple readouts
    },
    'optimizer': {
        'type': 'adam', # optimizer to be used
        'lr': {'base_lr': base_lr, # learning rate
               'warmup_epochs': 5 if args.start_from_epoch==0 else 2, # lr starts at base_lr/(lr scale factor) and scales up linearly for these many epochs
               'lr_scale_factor': 100 if args.start_from_epoch==0 else 1.5, # factor by which to scale the learning rate
               },
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs, # number of epochs (full cycle through the dataset)
        'device': 'cuda', # device to train the network on
        'dataloader': {
            'num_workers_train': args.num_workers, # number of cpu workers processing the batches 
            'prefetch_factor_train': 2, # reduced from 4 to limit system RAM usage
            'num_workers_val_test': 2, # do not need lots of workers for val/test
            'prefetch_factor_val_test': 2 
        }
    },
    'misc': {
        'use_amp': True, # use automatic mixed precision during training - forward pass .half(), backward full
        'batch_size_val_test': args.batch_size_val_test,
        'save_logs': 1, # after how many epochs should we save a copy of the logs
        'save_net': 1, # after how many epochs should we save a copy of the net - ensure this is a multiple of save_logs
        'start_from_epoch': args.start_from_epoch # at which epoch to start training (data pulled from epoch before that)
    }
}

hyp["gradient_checkpointing"] = bool(args.gradient_checkpointing)

hyp["dataset_mode"] = args.dataset_mode
if hyp["dataset_mode"] == 2:
    hyp["dataset"]["name"] = "cifar100"
elif hyp["dataset_mode"] == 1:
    hyp["dataset"]["name"] = "debug"

# --- set num_classes based on dataset ---
if hyp["dataset"]["name"] == "cifar100":
    hyp["dataset"]["n_classes"] = 100
elif hyp["dataset"]["name"] in ["ecoset", "miniecoset"]:
    hyp["dataset"]["n_classes"] = 565
elif hyp["dataset"]["name"] == "debug":
    # FakeData: n_classes muss zu deinem FakeData passen (z.B. 10 oder 100)
    hyp["dataset"]["n_classes"] = 100
else:
    raise ValueError(f"Unknown dataset name: {hyp['dataset']['name']}")

hyp["ecoset_debug_subset"] = args.ecoset_debug_subset
hyp["ecoset_debug_size"] = args.ecoset_debug_size

# Only enable MixUp/CutMix/RA for rn50 — BLT-VS training is unaffected
if args.network == 'rn50':
    hyp['augmentation'] = {
        'mixup_alpha': args.use_mixup,
        'cutmix_alpha': args.use_cutmix,
        'ra_reps': args.ra_reps,
    }
else:
    hyp['augmentation'] = {
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'ra_reps': 0,
    }
# -----------------------------
# Modular bottlenecks config
# -----------------------------
def parse_bottlenecks(s: str):
    s = (s or "").strip()
    if s == "":
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        edge, ch = item.split(":")
        out[edge.strip()] = int(ch.strip())
    return out

hyp["network"]["bottlenecks"] = parse_bottlenecks(args.bottlenecks)
print("Bottlenecks:", hyp["network"]["bottlenecks"])

# -----------------------------
# Create readable bottleneck string for folder name
# -----------------------------
if hyp["network"]["bottlenecks"]:
    bn_parts = []
    for k, v in hyp["network"]["bottlenecks"].items():
        clean_key = k.replace("->", "")
        bn_parts.append(f"{clean_key}-{v}")
    bottleneck_str = "_".join(bn_parts)
else:
    bottleneck_str = "none"

# Beispiel: V1->V2 aktiv mit 144 Channels
# hyp["network"]["bottlenecks"] = {"V1->V2": 144}
##################
### Training and evaluation
##################

print('\nAaaand it begins...\n')

def save_filtered_state_dict(state_dict, save_path): # because FLOP computation adds some keys to the state_dict which are not needed for saving
    # Get the model's state_dict
    state_dict = state_dict
    # Filter out keys containing 'total_ops' and 'total_params'
    filtered_state_dict = {k: v for k, v in state_dict.items() if not ('total_ops' in k or 'total_params' in k)}
    # Save the filtered state_dict
    torch.save(filtered_state_dict, save_path)

if __name__ == '__main__':

    print("\n==============================")
    print("DEBUG: NEW SCRIPT VERSION ACTIVE")
    print("==============================\n")

    areas_to_extract=["Retina","LGN","V1","V2","V3","V4","LOC"]

    # load the dataset loaders to iterate over for training and eval
    train_loader, val_loader, _, hyp = get_Dataset_loaders(hyp,['train','val'])

    print("Dataset mode:", hyp["dataset_mode"])
    #print("Number of classes:", hyp["dataset"]["n_classes"])
    print("Train dataset size:", len(train_loader.dataset))
    print("Number of train batches:", len(train_loader))


    net, net_name = get_network_model(hyp)
    print("Dataset n_classes:", hyp["dataset"]["n_classes"])
    if hasattr(net, "num_classes"):
        print("Model num_classes:", net.num_classes)
    net = net.float()
    # create the network
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    network_name = hyp["network"]["name"]
    dataset_name = hyp["dataset"]["name"]
    timesteps = hyp["network"]["timesteps"]

    if args.name:
        net_name = f"{args.name}__{timestamp}"
    else:
        net_name = (
            f"{network_name}__"
            f"{dataset_name}__"
            f"ts{timesteps}__"
            f"bn-{bottleneck_str}__"
            f"{timestamp}"
        )

    net = net.float()

    # creating folders for logging losses/acc and network weights
    log_path, net_path = create_folders_logging(net_name)
    print(f'Log_folders: {log_path} -- {net_path}')

    # ============================
    # Save model config and definition
    # ============================

    def _json_serializable(obj):
        """Convert non-serializable types so hyp can be saved as JSON."""
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return str(obj)

    # 1) Save hyperparameter config as JSON
    config_path = os.path.join(log_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hyp, f, indent=2, default=_json_serializable)
    print(f"Saved model config to {config_path}")

    # 2) Save command-line args as JSON
    args_path = os.path.join(log_path, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=_json_serializable)
    print(f"Saved command-line args to {args_path}")

    # 3) Save model architecture string (print(net) output)
    arch_path = os.path.join(log_path, "model_architecture.txt")
    with open(arch_path, "w") as f:
        f.write(str(net))
    print(f"Saved model architecture to {arch_path}")

    # 4) Save model class source code (if inspectable)
    model_src_path = os.path.join(log_path, "model_source.py")
    try:
        model_class = type(net)
        source_code = inspect.getsource(model_class)
        with open(model_src_path, "w") as f:
            f.write(f"# Source of {model_class.__module__}.{model_class.__name__}\n\n")
            f.write(source_code)
        print(f"Saved model source code to {model_src_path}")
    except (TypeError, OSError) as e:
        print(f"Could not save model source code: {e}")

    # 5) Save total parameter count
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    param_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }
    param_path = os.path.join(log_path, "param_count.json")
    with open(param_path, "w") as f:
        json.dump(param_info, f, indent=2)
    print(f"Saved parameter counts to {param_path}  (total={total_params:,}, trainable={trainable_params:,})")

    # Initialize network weights if not starting from scratch
    if hyp['misc']['start_from_epoch'] > 0:
        load_epoch = hyp['misc']['start_from_epoch']
        print(f'Loading epoch: {load_epoch}')
        net_save_path = f'{net_path}/{net_name}_epoch_{load_epoch}.pth'
        state_dict = torch.load(net_save_path)
        net.load_state_dict(state_dict)
        # load_filtered_state_dict(net, net_save_path)

    # Print the number of FLOPs for one pass
    if not args.network == 'blt_vnet':
        if args.bio_unroll == 1:
            print("\nSkipping FLOPs computation because bio_unroll=1 (thop/profile can break on this forward path).\n")
        else:
            dummy = torch.randn(1, 3, 128, 128) if args.network == 'vNet' else torch.randn(1, 3, 224, 224)
            print("\nFLOPs for one pass: {}\n".format(calculate_flops(net, dummy)))
        net.train()

    print(net)
    #print(net.bottlenecks)

    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print("\nLet's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(hyp['optimizer']['device'])

    # criterion and optimizer setup
    criterion = nn.CrossEntropyLoss(weight=hyp['dataset']['class_weights'], label_smoothing=0.1)
    hyp['optimizer']['type'] = args.optimizer_type
    optimizer = get_optimizer(hyp,net)
    scaler = torch.amp.GradScaler("cuda", enabled=hyp['misc']['use_amp']) # this is in service of mixed precision training

    # --- EMA model (rn50 only) ---
    ema_model = None
    if args.use_ema and args.network == 'rn50':
        from copy import deepcopy
        ema_model = deepcopy(net)
        ema_model.eval()
        ema_model.requires_grad_(False)
        ema_decay = args.ema_decay
        print(f'Using EMA with decay={ema_decay}')

    # --- LR Scheduler ---
    if args.lr_scheduler_type == 'cosine' and args.network == 'rn50':
        # Cosine annealing after warmup
        total_epochs = hyp['optimizer']['n_epochs']
        warmup_epochs = args.warmup_epochs
        # During warmup: linear ramp from base_lr/100 to base_lr
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0/100, end_factor=1.0, total_iters=warmup_epochs
        )
        # After warmup: cosine decay to 0
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        print(f'Using CosineAnnealing LR scheduler (warmup={warmup_epochs}, total={total_epochs})')
        use_cosine_scheduler = True
    else:
        # Default for BLT-VS: LinearFitScheduler with patience
        lr_scheduler = LinearFitScheduler(optimizer, num_epochs=5, factor=1./2, min_percent_change=1.0, mode='min', verbose=True, patience=args.lr_patience)
        # Warm-up scheduler - this already initialises the lr to base_lr/lr_scale_factor
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch_h: ((hyp['optimizer']['lr']['base_lr'] - hyp['optimizer']['lr']['base_lr']/hyp['optimizer']['lr']['lr_scale_factor']) / (hyp['optimizer']['lr']['warmup_epochs']-1)) * epoch_h + hyp['optimizer']['lr']['base_lr']/hyp['optimizer']['lr']['lr_scale_factor'])
        use_cosine_scheduler = False

    # logging losses and accuracies
    if hyp['misc']['start_from_epoch'] == 0:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_accuracies_all = []   # <-- NEU: timestep-wise val acc pro epoch
    else:
        log_data = np.load(log_path+'/loss_'+net_name+'.npz')
        train_losses = list(log_data['train_loss'][:hyp['misc']['start_from_epoch']])
        train_accuracies = list(log_data['train_accuracies'][:hyp['misc']['start_from_epoch']])
        val_losses = list(log_data['val_loss'][:hyp['misc']['start_from_epoch']])
        val_accuracies = list(log_data['val_accuracies'][:hyp['misc']['start_from_epoch']])

        # <-- NEU: falls vorhanden, sonst leere Liste
        if "val_accuracies_all" in log_data.files:
            val_accuracies_all = list(log_data["val_accuracies_all"][:hyp['misc']['start_from_epoch']])
        else:
            val_accuracies_all = []

    # saving the randomly initialized network
    if hyp['misc']['start_from_epoch'] == 0:
        if torch.cuda.device_count() > 1: # given how dataparallel works, we need to save the module's state_dict
            save_filtered_state_dict(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
            # torch.save(net.module.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
        else:
            save_filtered_state_dict(net.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')
            # torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{0}.pth')

    print('\nTraining begins here!\n')

    epoch = 1
    training_not_finished = 1

    best_val_acc = -float("inf")
    best_epoch = -1

    while training_not_finished: # Looping until we reach the desired number of epochs or convergence

        start = time.time()

        torch.cuda.synchronize()
        
        train_loss_running = 0.0
        train_acc_running = 0.0

        epoch_now = epoch+hyp['misc']['start_from_epoch']
        print('LR now: ',optimizer.param_groups[0]['lr'])

        # Update RA sampler epoch for proper shuffling
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_now)

        epoch_running_init_flag = 0

        # Reset memory stats
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            torch.cuda.reset_peak_memory_stats(device)
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_now}",
            leave=True,
            dynamic_ncols=True,
            file= sys.stdout
        )
        accum_steps = args.grad_accum_steps
        optimizer.zero_grad()
        for step_idx, (images, labels) in enumerate(pbar):

            imgs = images.to(hyp['optimizer']['device'])
            lbls = labels.to(hyp['optimizer']['device'])
            # For soft targets (MixUp/CutMix) keep float; for hard targets use long
            targets = lbls if lbls.dim() == 2 else lbls.long()
            hard_lbls = lbls.argmax(dim=1) if lbls.dim() == 2 else lbls
            # Move weights to the same device as inputs
            if criterion.weight is not None:
                criterion.weight = criterion.weight.to(imgs.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                if epoch == 1 and epoch_running_init_flag == 0:
                    print("Labels shape:", lbls.shape)
                    print("DEBUG: len(outputs) =", len(outputs))
                loss = criterion(outputs[0], targets) 
                if len(outputs) > 1:
                    for t in range(len(outputs)-1):
                        loss = loss + criterion(outputs[t+1], targets)
                loss = loss/len(outputs)
                loss = loss / accum_steps  # scale loss for accumulation
            
            scaler.scale(loss).backward()

            # --- Free computation graph and intermediate tensors immediately ---
            train_loss_running += loss.item() * accum_steps  # undo scaling for logging
            with torch.no_grad():
                current_acc = np.mean(compute_accuracy(outputs, hard_lbls))
            train_acc_running += current_acc
            n_timestep_outputs = len(outputs)  # save before del for use after loop
            del outputs, loss, imgs, lbls, targets, hard_lbls, images, labels

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                if args.grad_clipping:
                    scaler.unscale_(optimizer)
                    adaptive_gradient_clipping(net, clip_factor=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA update
                if ema_model is not None:
                    with torch.no_grad():
                        net_params = (net.module if hasattr(net, 'module') else net).state_dict()
                        for key, ema_param in ema_model.state_dict().items():
                            model_param = net_params[key]
                            ema_param.copy_(ema_decay * ema_param + (1.0 - ema_decay) * model_param)

            pbar.set_postfix({
                "loss": f"{train_loss_running / (step_idx + 1):.3f}",
                "acc": f"{current_acc:.2f}%"
            })

            # GPU utilization logging once per epoch (at step 10)
            if (step_idx + 1) == 10:
                for gpu_i in range(torch.cuda.device_count()):
                    mem_used = torch.cuda.memory_reserved(gpu_i) / (1024**3)
                    mem_total = torch.cuda.get_device_properties(gpu_i).total_memory / (1024**3)
                    print(f"  [Step {step_idx+1}] GPU {gpu_i}: {mem_used:.1f}/{mem_total:.1f} GB reserved")

            if epoch_running_init_flag == 0:
                epoch_running_init_flag = 1
        pbar.close()

        train_losses.append(train_loss_running/len(train_loader))
        train_accuracies.append(train_acc_running/len(train_loader))

        max_mem_allocated = 0
        gpu_count = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            max_mem_allocated += torch.cuda.max_memory_reserved(device) / (1024**3)
        print(f'Max GPU(s) memory reserved: {max_mem_allocated} Gb; {gpu_count} GPU(s)')
        
        # Use EMA model for validation if available
        eval_model = ema_model if ema_model is not None else net
        eval_model.eval()
        val_loss_running, val_acc_running = eval_network(val_loader, eval_model, criterion, hyp)
        net.train()

        # val_acc_running sollte timestep-wise sein -> z.B. [acc_t1, acc_t2, ...]
        val_acc_running = val_acc_running / len(val_loader)

        print("DEBUG type(val_acc_running):", type(val_acc_running))
        print("DEBUG val_acc_running:", val_acc_running)

        # Speichere timestep-wise acc pro epoch (als numpy array)
        val_acc_ts = np.array(val_acc_running, dtype=float)
        val_accuracies_all.append(val_acc_ts)
        print("DEBUG val_accuracies_all shape so far:", np.array(val_accuracies_all).shape)

        # Zusätzlich wie bisher: mean accuracy pro epoch
        val_losses.append(val_loss_running / len(val_loader) / n_timestep_outputs)
        val_accuracies.append(float(np.mean(val_acc_ts)))

        current_val_acc = val_accuracies[-1]
        epoch_save = epoch + hyp['misc']['start_from_epoch']

        if current_val_acc > best_val_acc:

            best_val_acc = current_val_acc
            best_epoch = epoch_save

            print(f"New BEST model at epoch {best_epoch} (val acc = {best_val_acc:.2f}%)")

            # Save EMA model if available, otherwise save raw model
            best_state_dict = ema_model.state_dict() if ema_model is not None else \
                (net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict())
            save_filtered_state_dict(best_state_dict, f'{net_path}/{net_name}_BEST.pth')


        ts_string = " | ".join([f"t{i+1}:{acc:.2f}%" for i, acc in enumerate(val_acc_ts)])
        print(f"Val acc per timestep → {ts_string}")

        print('Epoch time: ', "{:.2f}".format(time.time() - start), ' seconds')

        print(f'Train loss: {train_losses[-1]:.2f}; acc: {train_accuracies[-1]:.2f}%')
        print(f'Val loss: {val_losses[-1]:.2f}; acc: {val_accuracies[-1]:.2f}%; acc_t: {val_acc_running}')

        if (epoch) < hyp['optimizer']['lr']['warmup_epochs']:
            if use_cosine_scheduler:
                lr_scheduler.step()
            else:
                warmup_scheduler.step()
        else:
            if use_cosine_scheduler:
                lr_scheduler.step()
            else:
                lr_scheduler.step(val_losses[-1])

        if (epoch+hyp['misc']['start_from_epoch']) % hyp['misc']['save_logs'] == 0:
            print('Saving metrics!')
            np.savez(
                log_path + '/loss_' + net_name + '.npz',
                train_loss=train_losses,
                val_loss=val_losses,
                train_accuracies=train_accuracies,
                val_accuracies=val_accuracies,
                val_accuracies_all=np.array(val_accuracies_all, dtype=float)
            )

        epoch += 1
        if hyp['optimizer']['n_epochs'] > 0:
            if epoch > hyp['optimizer']['n_epochs']:
                training_not_finished = 0
                print('\n Done training! - #epochs completed\n')
        elif hyp['optimizer']['n_epochs'] == -1:
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                training_not_finished = 0
                print('\n Done training! - LR reached 1e-6 i.e. converged\n')

    final_epoch = epoch + hyp['misc']['start_from_epoch'] - 1
    print(f"\nSaving LAST checkpoint (epoch {final_epoch})")

    if torch.cuda.device_count() > 1:
            save_filtered_state_dict(
                net.module.state_dict(),
                f'{net_path}/{net_name}_LAST.pth'
            )
    else:
            save_filtered_state_dict(
                net.state_dict(),
                f'{net_path}/{net_name}_LAST.pth'
            )

    # getting test loss and acc
    _, _, test_loader, hyp = get_Dataset_loaders(hyp,['test'])
    net.eval()
    if test_loader is not None:
        test_loss_running, test_acc_running = eval_network(test_loader, net, criterion, hyp)
        test_acc = test_acc_running / len(test_loader)
        print("Test acc:", test_acc)
    else:
        print("Skipping test evaluation (no test loader in debug mode)")
    if test_loader is not None:
        print(f'Test accuracies over time (%): {test_acc}')
        print('Saving metrics!')

        np.savez(
            log_path + '/loss_' + net_name + '.npz',
            train_loss=train_losses,
            val_loss=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            val_accuracies_all=np.array(val_accuracies_all, dtype=float),
            test_accuracies=test_acc
        )

    else:
        print('Saving metrics!')

        np.savez(
            log_path + '/loss_' + net_name + '.npz',
            train_loss=train_losses,
            val_loss=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            val_accuracies_all=np.array(val_accuracies_all, dtype=float)
        )

    # ============================
    # Streaming PCA via covariance accumulation
    # (only for models that support extract_actvs, e.g. BLT_VS)
    # ============================

    if network_name != 'blt_vs':
        print(f"\nSkipping PCA extraction (not supported for network '{network_name}').")
    else:
        print("\nExtracting PCA statistics (streaming, no activation saving)...")
        _, val_loader, _, hyp = get_Dataset_loaders(hyp, ['val'])
        print("Validation batches for PCA:", len(val_loader))

        areas_to_extract = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC"]
        timesteps_to_extract = list(range(hyp["network"]["timesteps"]))

        cov_mats = {}
        sum_vecs = {}
        counts = {}

        extract_batches = 0
        max_extract_batches = 50   # increase later if you want more stable PCA

        model_for_extract = net.module if isinstance(net, nn.DataParallel) else net
        model_for_extract.eval()

        #Setup for unexpected values in layers
        first_signal = compute_first_signal(hyp["network"]["bottlenecks"], hyp["network"]["skip_connections"])
        print(f"Computed first_signal based on connections: {first_signal}")
        
        threshold = 1e-8

        with torch.no_grad():
            for images, labels in val_loader:

                imgs = images.to(hyp['optimizer']['device'])

                outputs, activations = model_for_extract(
                    imgs,
                    extract_actvs=True,
                    areas=areas_to_extract,
                    timesteps=timesteps_to_extract
                ) 

                for area in activations:
                    for t in activations[area]:

                        act = activations[area][t]

                        if act is None:
                            continue

                        if isinstance(act, dict):
                            act = next(iter(act.values()))

                        # check activations before signal arrival
                        if area in first_signal and t < first_signal[area]:

                            max_val = act.abs().max().item()
                            mean_val = act.abs().mean().item()

                            if extract_batches == 0:
                                print(f"{area} t{t}: max={max_val:.2e}, mean={mean_val:.2e}")

                            if max_val > threshold:
                                print(f"⚠ Unexpected large activation at {area} t{t}")

                            continue

                        key = f"{area}_t{t}"

                        # Optional spatial subsampling to reduce compute
                        act = act[:, :, ::2, ::2]

                        B, C, H, W = act.shape

                        # reshape to (N, C), where N = B * H * W
                        X = act.permute(0, 2, 3, 1).reshape(-1, C)

                        # use float32 for stable covariance accumulation
                        X = X.detach().float()

                        if key not in cov_mats:
                            cov_mats[key] = torch.zeros(C, C, device=X.device, dtype=torch.float32)
                            sum_vecs[key] = torch.zeros(C, device=X.device, dtype=torch.float32)
                            counts[key] = 0

                        cov_mats[key] += X.T @ X
                        sum_vecs[key] += X.sum(dim=0)
                        counts[key] += X.shape[0]

                extract_batches += 1
                if extract_batches >= max_extract_batches:
                    break

        print("Finished accumulating covariance matrices.")

        # ============================
        # Compute PCA results
        # ============================

        pca_results = {}

        for key in cov_mats:

            n = counts[key]

            mean = sum_vecs[key] / n

            cov = (cov_mats[key] / n) - torch.outer(mean, mean)

            cov = cov.cpu().numpy()

            eigvals, eigvecs = np.linalg.eigh(cov)

            # sort descending
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]

            # numerical safety
            eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

            total_var = eigvals.sum()
            if total_var <= 0:
                explained = np.zeros_like(eigvals)
            else:
                explained = eigvals / total_var

            cumulative = np.cumsum(explained)

            channels_90 = int(np.searchsorted(cumulative, 0.90) + 1)
            channels_95 = int(np.searchsorted(cumulative, 0.95) + 1)
            channels_99 = int(np.searchsorted(cumulative, 0.99) + 1)

            pca_results[f"{key}_eigvals"] = eigvals
            pca_results[f"{key}_explained"] = explained
            pca_results[f"{key}_cumulative"] = cumulative
            pca_results[f"{key}_channels_90"] = np.array([channels_90])
            pca_results[f"{key}_channels_95"] = np.array([channels_95])
            pca_results[f"{key}_channels_99"] = np.array([channels_99])

            print(
                f"{key}: "
                f"90%={channels_90}, "
                f"95%={channels_95}, "
                f"99%={channels_99}, "
                f"total_channels={len(eigvals)}"
            )

        pca_path = log_path + "/pca_results_streaming.npz"
        np.savez(pca_path, **pca_results)

        print("Saved PCA results to:", pca_path)
    
    if hyp["dataset_mode"] != 1:

        print("Saving training plots (annotated + summary)...")

        from matplotlib.ticker import MaxNLocator
        import pandas as pd

        epochs = np.arange(1, len(train_losses) + 1)

        train_loss = np.array(train_losses)
        val_loss = np.array(val_losses)
        train_acc = np.array(train_accuracies)
        val_acc = np.array(val_accuracies)

        # ============================
        # Compute best values
        # ============================

        best_val_acc = np.max(val_acc)
        best_val_epoch = np.argmax(val_acc) + 1
        train_acc_at_best = train_acc[best_val_epoch - 1]

        best_val_loss = np.min(val_loss)
        best_loss_epoch = np.argmin(val_loss) + 1
        train_loss_at_best = train_loss[best_loss_epoch - 1]

        # ============================
        # ACCURACY PLOT (Annotated)
        # ============================

        plt.figure(figsize=(8,5))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")

        plt.scatter(best_val_epoch, best_val_acc, color='red', zorder=5)
        plt.axvline(best_val_epoch, linestyle='--', alpha=0.5)

        gap = train_acc_at_best - best_val_acc

        plt.annotate(
            f"Best Val Acc: {best_val_acc:.2f}%\nEpoch {best_val_epoch}\nGap: {gap:.2f}%",
            (best_val_epoch, best_val_acc),
            textcoords="offset points",
            xytext=(-60,20)
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve (Annotated)")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path + "/accuracy_plot.png", dpi=300)
        plt.close()

        # ============================
        # LOSS PLOT (Annotated)
        # ============================

        plt.figure(figsize=(8,5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")

        plt.scatter(best_loss_epoch, best_val_loss, color='green', zorder=5)
        plt.axvline(best_loss_epoch, linestyle='--', alpha=0.5)

        plt.annotate(
            f"Lowest Val Loss: {best_val_loss:.4f}\nEpoch {best_loss_epoch}",
            (best_loss_epoch, best_val_loss),
            textcoords="offset points",
            xytext=(-60,-25)
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve (Annotated)")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path + "/loss_plot.png", dpi=300)
        plt.close()

        # ============================
        # SUMMARY TABLE
        # ============================

        summary = pd.DataFrame({
            "Metric": [
                "Best Val Accuracy (%)",
                "Train Accuracy @ Best Val Epoch (%)",
                "Validation Loss @ Best Epoch",
                "Train Loss @ Best Val Epoch"
            ],
            "Value": [
                round(best_val_acc, 3),
                round(train_acc_at_best, 3),
                round(best_val_loss, 4),
                round(train_loss_at_best, 4)
            ]
        })

        fig, ax = plt.subplots(figsize=(7,2))
        ax.axis('off')
        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.savefig(log_path + "/training_summary_table.png", dpi=300)
        plt.close()

        # ============================
        # TIMESTEP ACCURACY PLOT (Best Val Epoch)
        # ============================

        if len(val_accuracies_all) > 0:
            best_epoch_idx = int(np.argmax(val_accuracies))  # index in [0..n_epochs-1]
            best_epoch = best_epoch_idx + 1

            ts_acc = np.array(val_accuracies_all[best_epoch_idx], dtype=float)
            timesteps = np.arange(1, len(ts_acc) + 1)

            plt.figure(figsize=(7,4))
            plt.plot(timesteps, ts_acc, marker="o")
            plt.xlabel("Timestep")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Validation Accuracy over Timesteps (Best Epoch {best_epoch})")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(log_path + "/val_accuracy_over_timesteps_best_epoch.png", dpi=300)
            plt.close()

        # ============================
        # TIMESTEP ACCURACY PLOT (5 evenly spaced epochs)
        # ============================

        if len(val_accuracies_all) > 0:
            N = len(val_accuracies_all)

            # pick 5 epochs evenly spaced, like: 10->2,4,6,8,10 and 20->4,8,12,16,20
            if N >= 5:
                selected_epochs = np.linspace(1, N, 6)[1:]          # 5 values, excludes 1st
                selected_epochs = np.unique(np.rint(selected_epochs).astype(int))
                selected_epochs = np.clip(selected_epochs, 1, N)
                selected_epochs = np.unique(selected_epochs)         # in case rounding causes duplicates

                # If rounding collapsed duplicates (rare), fall back to evenly spaced integers
                if len(selected_epochs) < 5:
                    selected_epochs = np.unique(np.linspace(1, N, 5).astype(int))
            else:
                selected_epochs = np.arange(1, N + 1)

            plt.figure(figsize=(8,5))

            for ep in selected_epochs:
                ts_acc = np.array(val_accuracies_all[ep - 1], dtype=float)
                timesteps = np.arange(1, len(ts_acc) + 1)
                plt.plot(timesteps, ts_acc, marker="o", label=f"Epoch {ep}")

            plt.xlabel("Timestep")
            plt.ylabel("Validation Accuracy (%)")
            plt.title("Validation Accuracy over Timesteps (5 checkpoints)")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(log_path + "/val_accuracy_over_timesteps_5epochs.png", dpi=300)
            plt.close()

        # ============================
        # RECURRENCE ANALYSIS PLOTS
        # ============================

        loss_file = log_path + '/loss_' + net_name + '.npz'

        if os.path.exists(loss_file):

            data = np.load(loss_file)

            if "val_accuracies_all" in data.files:

                val_all = data["val_accuracies_all"]

                epochs, timesteps = val_all.shape
                last_epoch = val_all[-1]

                t1 = last_epoch[0]
                tmax = last_epoch.max()
                rec_score = tmax - t1
                percent_gain = (rec_score / t1) * 100 if t1 != 0 else 0

                print(f"Final Epoch: {epochs}")
                print(f"T1 Accuracy: {t1:.2f}")
                print(f"Tmax Accuracy: {tmax:.2f}")
                print(f"Recurrence Score (Δ): {rec_score:.2f}")
                print(f"Relative Gain: {percent_gain:.2f}%")

                # --------------------------
                # Timestep curve (final epoch)
                # --------------------------

                plt.figure()
                plt.plot(range(1, timesteps+1), last_epoch, marker="o")
                plt.xlabel("Timestep")
                plt.ylabel("Validation Accuracy (%)")
                plt.title("Validation Accuracy over Timesteps (Final Epoch)")
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(os.path.join(log_path, "timestep_curve_last_epoch.png"), dpi=300)
                plt.close()

                # --------------------------
                # Table over epochs
                # --------------------------

                columns = [f"t{i+1}" for i in range(timesteps)] + ["t_max", "Δ (tmax-t1)", "Δ (%)"]

                rows = []

                for e in range(epochs):

                    row = val_all[e]

                    t1 = row[0]
                    tmax = row.max()
                    delta = tmax - t1
                    percent = (delta / t1) * 100 if t1 != 0 else 0

                    full_row = list(np.round(row, 2))
                    full_row += [round(tmax, 2), round(delta, 2), round(percent, 2)]

                    rows.append(full_row)

                fig, ax = plt.subplots(figsize=(14, 0.4 * epochs + 2))
                ax.axis("off")

                table = ax.table(
                    cellText=rows,
                    colLabels=columns,
                    rowLabels=[f"E{e+1}" for e in range(epochs)],
                    loc="center"
                )

                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.2)

                plt.title("Validation Accuracy – All Epochs (Timestep Summary)", pad=20)
                plt.tight_layout()

                plt.savefig(os.path.join(log_path, "timestep_table.png"), dpi=300, bbox_inches="tight")
                plt.close()

                # --------------------------
                # Recurrence gain heatmap
                # --------------------------

                gain = val_all - val_all[:,0:1]

                plt.figure(figsize=(10,6))
                im = plt.imshow(gain, aspect="auto", cmap="viridis")

                plt.colorbar(im, label="Gain relative to t1 (Accuracy %)")
                plt.xlabel("Timestep")
                plt.ylabel("Epoch")
                plt.title("Recurrence Gain over Training (Relative to t1)")
                plt.xticks(range(timesteps), [f"t{i+1}" for i in range(timesteps)])
                plt.yticks(range(0, epochs, max(1, epochs//10)))

                plt.tight_layout()

                plt.savefig(os.path.join(log_path, "recurrence_gain_heatmap.png"), dpi=300)
                plt.close()

        else:
            print("Loss file not found, skipping recurrence plots.")


        # ============================
        # PCA DIMENSIONALITY PLOTS
        # ============================

        pca_path = log_path + "/pca_results_streaming.npz"
        if os.path.exists(pca_path):

            import matplotlib.pyplot as plt
            import numpy as np
            import os

            data = np.load(pca_path)

            areas = ["Retina","LGN","V1","V2","V3","V4","LOC"]
            timesteps = hyp["network"]["timesteps"]

            total_channels = {
                "Retina":32,
                "LGN":32,
                "V1":576,
                "V2":480,
                "V3":352,
                "V4":256,
                "LOC":352
            }

            levels = [90,95,99]

            for level in levels:

                dim_matrix = []

                for area in areas:

                    row = []

                    for t in range(timesteps):

                        key = f"{area}_t{t}_channels_{level}"

                        if key in data:
                            row.append(data[key][0])
                        else:
                            row.append(0)

                    row.append(total_channels[area])
                    dim_matrix.append(row)

                dim_matrix = np.array(dim_matrix)

                heatmap_abs = dim_matrix[:, :-1]

                totals = np.array([total_channels[a] for a in areas])[:, None]
                heatmap_rel = heatmap_abs / totals

                fig, axes = plt.subplots(
                    2, 2,
                    figsize=(22,10),
                    gridspec_kw={
                        "height_ratios":[1,0.65],
                        "wspace":0.35,
                        "hspace":0.12
                    }
                )

                # ---------------------------
                # Absolute heatmap
                # ---------------------------

                ax = axes[0,0]

                im = ax.imshow(heatmap_abs, aspect="auto")

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f"Channels for {level}% variance")

                ax.set_xticks(range(timesteps))
                ax.set_xticklabels(range(timesteps))
                ax.set_yticks(range(len(areas)))
                ax.set_yticklabels(areas)

                ax.set_xlabel("Timestep")
                ax.set_ylabel("Visual Area")
                ax.set_title(f"Representation Dimensionality ({level}% variance)")


                # ---------------------------
                # Relative heatmap
                # ---------------------------

                ax = axes[0,1]

                im = ax.imshow(heatmap_rel, aspect="auto", vmin=0, vmax=1)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Fraction of total channels")

                ax.set_xticks(range(timesteps))
                ax.set_xticklabels(range(timesteps))
                ax.set_yticks(range(len(areas)))
                ax.set_yticklabels(areas)

                ax.set_xlabel("Timestep")
                ax.set_ylabel("Visual Area")
                ax.set_title(f"Relative Dimensionality ({level}% variance)")


                # ---------------------------
                # Absolute table
                # ---------------------------

                ax = axes[1,0]
                ax.axis("off")

                table_abs = ax.table(
                    cellText=dim_matrix,
                    rowLabels=areas,
                    colLabels=[f"t{i}" for i in range(timesteps)] + ["Total"],
                    cellLoc="center",
                    bbox=[0,0.20,1,0.75]
                )

                table_abs.auto_set_font_size(False)
                table_abs.set_fontsize(11)
                table_abs.scale(1.2, 1.6)


                # ---------------------------
                # Relative table
                # ---------------------------

                ax = axes[1,1]
                ax.axis("off")

                rel_matrix = np.round(heatmap_rel * 100, 1)

                rel_matrix = np.concatenate(
                    [rel_matrix, np.full((len(areas),1), 100)],
                    axis=1
                )

                table_rel = ax.table(
                    cellText=rel_matrix,
                    rowLabels=areas,
                    colLabels=[f"t{i}" for i in range(timesteps)] + ["Total"],
                    cellLoc="center",
                    bbox=[0,0.20,1,0.75]
                )

                table_rel.auto_set_font_size(False)
                table_rel.set_fontsize(11)
                table_rel.scale(1.2, 1.6)


                plt.subplots_adjust(
                    left=0.06,
                    right=0.96,
                    top=0.92,
                    bottom=0.05
                )

                save_path = os.path.join(log_path, f"pca_dimensionality_{level}.png")

                plt.savefig(save_path, dpi=300, bbox_inches="tight")

                plt.close(fig)

            print("PCA plots saved.")

        else:
            print("PCA results not found, skipping PCA plots.")

        print("Annotated plots and summary table saved successfully.")

    else:
        print("Skipping plot saving (debug dataset mode).")

    