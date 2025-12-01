# run_threshold_sweep_with_logs.py
import os
import subprocess
import sys
from datetime import datetime

def run_threshold_experiments():
    """Testing with different thresholds and logging results."""
    
    thresholds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]
    results_summary = []
    
    base_config = "configs/test_potato/yolo_world_v2_x_potato_test.py"
    checkpoint = "weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
    # checkpoint = "weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth"
    # checkpoint = "weights/yolo_world_v2_m_obj365v1_goldg_pretrain_1280ft-77d0346d.pth"
    model_type = "yolo_world_v2_x"
    
    # Create log directory
    log_dir = f"test_logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    for threshold in thresholds:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*50}")
        print(f"Running test with threshold: {threshold}")
        print(f"{'='*50}")
        
        work_dir = f"work_dirs/potato_test/{model_type}/threshold_{threshold:.3f}"
        os.makedirs(work_dir, exist_ok=True)
        
        # File to save logs
        log_file = f"{log_dir}/threshold_{threshold:.3f}_{timestamp}.log"
        
        cmd = [
            'python', 'tools/test.py',
            base_config,
            checkpoint,
            '--work-dir', work_dir,
            '--out', f'{work_dir}/metrics/results.pkl',
            '--show-dir', 'visualizations',
            '--cfg-options', f'model.test_cfg.score_thr={threshold}'
        ]
        
        try:
            print(f"Starting test... Logs: {log_file}")
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Save logs and print to console
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    print(f"Successfully completed threshold {threshold}")
                    results_summary.append({
                        'threshold': threshold,
                        'work_dir': work_dir,
                        'log_file': log_file,
                        'status': 'success'
                    })
                else:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                    
        except Exception as e:
            print(f"Failed for threshold {threshold}: {e}")
            results_summary.append({
                'threshold': threshold,
                'work_dir': work_dir,
                'log_file': log_file,
                'status': 'failed',
                'error': str(e)
            })
    
    print(f"\n{'='*50}")
    print(results_summary)
    print(f"\n{'='*50}")
    print("Threshold sweep completed!")
    
    return results_summary

if __name__ == '__main__':
    run_threshold_experiments()