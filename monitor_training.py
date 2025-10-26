import os
import glob
import time
from pathlib import Path
import re

def parse_log_file(log_path):
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    info = {
        'total_lines': len(lines),
        'last_epoch': None,
        'last_loss': None,
        'latest_update': None
    }
    
    for line in reversed(lines[-100:]):
        if 'Epoch:' in line and 'Loss:' in line:
            epoch_match = re.search(r'Epoch:\s*(\d+)', line)
            loss_match = re.search(r'Loss:\s*([\d.]+)', line)
            
            if epoch_match and not info['last_epoch']:
                info['last_epoch'] = int(epoch_match.group(1))
            if loss_match and not info['last_loss']:
                info['last_loss'] = float(loss_match.group(1))
            
            if info['last_epoch'] and info['last_loss']:
                break
    
    return info

def monitor_training():
    print("=" * 80)
    print("GAZEGAUSSIAN TRAINING MONITOR")
    print("=" * 80)
    print("\nPress Ctrl+C to stop monitoring")
    print("-" * 80)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 80)
            print("TRAINING STATUS - {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            print("=" * 80)
            
            work_dirs = Path("./work_dirs")
            if not work_dirs.exists():
                print("\n⏳ No training started yet")
                print("   Waiting for work_dirs to be created...")
                time.sleep(5)
                continue
            
            subdirs = [d for d in work_dirs.iterdir() if d.is_dir()]
            
            if not subdirs:
                print("\n⏳ Training initializing...")
                time.sleep(5)
                continue
            
            for subdir in subdirs:
                print(f"\n📁 {subdir.name}")
                print("-" * 80)
                
                checkpoints = list(subdir.glob("**/*.pth"))
                if checkpoints:
                    latest_ckpt = max(checkpoints, key=os.path.getctime)
                    ckpt_time = time.ctime(os.path.getctime(latest_ckpt))
                    print(f"   Latest checkpoint: {latest_ckpt.name}")
                    print(f"   Created: {ckpt_time}")
                else:
                    print("   No checkpoints yet")
                
                log_files = list(subdir.glob("**/*.log"))
                if log_files:
                    latest_log = max(log_files, key=os.path.getctime)
                    log_info = parse_log_file(latest_log)
                    
                    if log_info:
                        print(f"   Log file: {latest_log.name} ({log_info['total_lines']} lines)")
                        if log_info['last_epoch']:
                            print(f"   Current epoch: {log_info['last_epoch']}")
                        if log_info['last_loss']:
                            print(f"   Latest loss: {log_info['last_loss']:.6f}")
                else:
                    print("   No logs yet")
            
            gpu_info_available = False
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"\n🖥️  GPU STATUS")
                    print("-" * 80)
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"   GPU {i}: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB used")
                    gpu_info_available = True
            except:
                pass
            
            if not gpu_info_available:
                print(f"\n💡 GPU info not available (torch not imported in this process)")
            
            print("\n" + "=" * 80)
            print("Refreshing in 10 seconds... (Ctrl+C to stop)")
            print("=" * 80)
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        print("=" * 80)

if __name__ == "__main__":
    monitor_training()
