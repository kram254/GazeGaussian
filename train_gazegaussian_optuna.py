import argparse
import random
import os
import torch
import numpy as np
import optuna
from optuna.trial import TrialState
from configs.gazegaussian_options import BaseOptions
from trainer.gazegaussian_trainer import GazeGaussianTrainer
from utils.recorder import GazeGaussianTrainRecorder


def auto_argparse_from_class(cls_instance):
    parser = argparse.ArgumentParser(description="Auto argparse from class")
    
    for attribute, value in vars(cls_instance).items():
        if isinstance(value, bool):
            parser.add_argument(f'--{attribute}', action='store_true' if not value else 'store_false',
                                help=f"Flag for {attribute}, default is {value}")
        elif isinstance(value, list):
            parser.add_argument(f'--{attribute}', type=type(value[0]), nargs='+', default=value,
                                help=f"List for {attribute}, default is {value}")
        else:
            parser.add_argument(f'--{attribute}', type=type(value), default=value,
                                help=f"Argument for {attribute}, default is {value}")

    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of Optuna trials to run')
    parser.add_argument('--optuna_storage', type=str, default='sqlite:///gazegaussian_optuna.db',
                        help='Optuna storage database path')
    parser.add_argument('--study_name', type=str, default='gazegaussian_study',
                        help='Optuna study name')
    parser.add_argument('--optuna_timeout', type=int, default=None,
                        help='Timeout in seconds for optimization')

    return parser


def suggest_hyperparameters(trial, base_opt):
    opt = BaseOptions()
    
    for attr, value in vars(base_opt).items():
        setattr(opt, attr, value)
    
    opt.lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    
    opt.dit_depth = trial.suggest_categorical('dit_depth', [4, 6, 8, 12])
    opt.dit_num_heads = trial.suggest_categorical('dit_num_heads', [4, 8, 16])
    opt.dit_patch_size = trial.suggest_categorical('dit_patch_size', [4, 8, 16])
    
    opt.vgg_importance = trial.suggest_float('vgg_importance', 0.05, 0.5, log=True)
    opt.eye_loss_importance = trial.suggest_float('eye_loss_importance', 0.5, 2.0)
    opt.gaze_loss_importance = trial.suggest_float('gaze_loss_importance', 0.05, 0.5, log=True)
    opt.orthogonality_loss_importance = trial.suggest_float('orthogonality_loss_importance', 0.001, 0.1, log=True)
    
    opt.eye_lr_mult = trial.suggest_float('eye_lr_mult', 0.5, 2.0)
    
    trial_name = f"{base_opt.name}_trial_{trial.number}"
    opt.name = trial_name
    
    return opt


class OptunaCallback:
    def __init__(self, trial):
        self.trial = trial
        self.best_val_loss = float('inf')
        
    def __call__(self, epoch, val_loss):
        self.trial.report(val_loss, epoch)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial, base_opt, train_data_loader, valid_data_loader):
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    random.seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    opt = suggest_hyperparameters(trial, base_opt)
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"{'='*80}")
    print(f"  Learning Rate: {opt.lr:.6f}")
    print(f"  DiT Depth: {opt.dit_depth}")
    print(f"  DiT Num Heads: {opt.dit_num_heads}")
    print(f"  DiT Patch Size: {opt.dit_patch_size}")
    print(f"  VGG Importance: {opt.vgg_importance:.4f}")
    print(f"  Eye Loss Importance: {opt.eye_loss_importance:.4f}")
    print(f"  Gaze Loss Importance: {opt.gaze_loss_importance:.4f}")
    print(f"  Orthogonality Loss Importance: {opt.orthogonality_loss_importance:.4f}")
    print(f"  Eye LR Multiplier: {opt.eye_lr_mult:.4f}")
    print(f"{'='*80}\n")
    
    recorder = GazeGaussianTrainRecorder(opt)
    trainer = GazeGaussianTrainer(opt, recorder)
    
    callback = OptunaCallback(trial)
    
    try:
        best_val_loss = trainer.train_with_optuna(
            train_data_loader=train_data_loader,
            n_epochs=opt.num_epochs,
            valid_data_loader=valid_data_loader,
            optuna_callback=callback
        )
        
        return best_val_loss
        
    except optuna.TrialPruned:
        print(f"\n⚠️  Trial {trial.number} pruned at epoch {callback.trial.last_step}")
        raise
    
    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed with error: {str(e)}")
        raise


def main():
    base_options = BaseOptions()
    parser = auto_argparse_from_class(base_options)
    args = parser.parse_args()
    
    from dataloader.eth_xgaze import get_train_loader, get_val_loader
    
    print("\n" + "="*80)
    print("GAZEGAUSSIAN OPTUNA OPTIMIZATION")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Data Directory: {args.img_dir}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Study Name: {args.study_name}")
    print(f"Storage: {args.optuna_storage}")
    print("="*80 + "\n")
    
    print("Loading training data...")
    train_data_loader = get_train_loader(
        args, 
        data_dir=args.img_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        evaluate="landmark", 
        is_shuffle=True, 
        dataset_name=args.dataset_name
    )
    print(f"✓ Training samples: {len(train_data_loader.dataset)}")
    
    print("Loading validation data...")
    valid_data_loader = get_val_loader(
        args,
        data_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        evaluate="landmark",
        dataset_name=args.dataset_name
    )
    print(f"✓ Validation samples: {len(valid_data_loader.dataset)}")
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=pruner,
        storage=args.optuna_storage,
        load_if_exists=True
    )
    
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    print(f"Pruner: MedianPruner")
    print(f"  - Startup trials: 3 (no pruning)")
    print(f"  - Warmup steps: 5 (epochs before pruning)")
    print(f"  - Check interval: every epoch")
    print("="*80 + "\n")
    
    study.optimize(
        lambda trial: objective(trial, args, train_data_loader, valid_data_loader),
        n_trials=args.n_trials,
        timeout=args.optuna_timeout,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    
    print(f"\nStudy statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed: {len(complete_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Failed: {len(failed_trials)}")
    
    if len(complete_trials) > 0:
        print(f"\n🎯 Best trial:")
        trial = study.best_trial
        print(f"  Value (validation loss): {trial.value:.6f}")
        print(f"  Trial number: {trial.number}")
        print(f"\n  Best hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        best_checkpoint_dir = f"{args.logdir}/{args.name}_trial_{trial.number}/checkpoints"
        print(f"\n  Best checkpoint location:")
        print(f"    {best_checkpoint_dir}")
        
        results_dir = "optuna_results"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        best_params_path = os.path.join(results_dir, "best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump(trial.params, f, indent=2)
        print(f"\n  Saved best hyperparameters to: {best_params_path}")
        
        try:
            import optuna.visualization as vis
            import plotly.io as pio
            
            print("\n📊 Generating visualizations...")
            
            fig = vis.plot_optimization_history(study)
            fig.write_html(os.path.join(results_dir, "optimization_history.html"))
            print(f"  ✓ Optimization history: {results_dir}/optimization_history.html")
            
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(results_dir, "param_importances.html"))
            print(f"  ✓ Parameter importances: {results_dir}/param_importances.html")
            
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(results_dir, "parallel_coordinate.html"))
            print(f"  ✓ Parallel coordinate: {results_dir}/parallel_coordinate.html")
            
            fig = vis.plot_slice(study)
            fig.write_html(os.path.join(results_dir, "slice_plot.html"))
            print(f"  ✓ Slice plot: {results_dir}/slice_plot.html")
            
        except Exception as e:
            print(f"\n⚠️  Could not generate visualizations: {str(e)}")
            print("  Install plotly: pip install plotly kaleido")
    else:
        print("\n❌ No trials completed successfully!")
    
    print("\n" + "="*80)
    print("To view interactive dashboard, run:")
    print(f"  optuna-dashboard {args.optuna_storage}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
