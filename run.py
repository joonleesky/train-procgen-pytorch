import subprocess
from multiprocessing import Pool

if __name__=='__main__':
    experiments = [
        {'--exp_name': 'easy-run-all',
         '--param_name': 'easy',
         '--start_level': '0',
         '--num_levels': '0',
         '--distribution_mode': 'easy',
         '--gpu_device': '0',
         '--num_timesteps': '25000000'},
        {'--exp_name': 'easy-run-200',
         '--param_name': 'easy-200',
         '--start_level': '0',
         '--num_levels': '200',
         '--distribution_mode': 'easy',
         '--gpu_device': '1',
         '--num_timesteps': '25000000'},
        {'--exp_name': 'hard-run-all',
         '--param_name': 'hard',
         '--start_level': '0',
         '--num_levels': '0',
         '--distribution_mode': 'hard',
         '--gpu_device': '2',
         '--num_timesteps': '200000000'},
        {'--exp_name': 'hard-run-500',
         '--param_name': 'hard-500',
         '--start_level': '0',
         '--num_levels': '500',
         '--distribution_mode': 'hard',
         '--gpu_device': '3',
         '--num_timesteps': '200000000'},
    ]
    def run_experiment(experiment):
        cmd = ['python', 'train.py']
        print(experiment)
        for key, value in experiment.items():
            cmd.append(key)
            cmd.append(value)
        return subprocess.call(cmd)

    pool = Pool(4)
    pool.map(run_experiment, experiments)
    pool.close()
