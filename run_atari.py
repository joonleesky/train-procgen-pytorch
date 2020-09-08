import subprocess
from multiprocessing import Pool

if __name__=='__main__':
    experiments = [
        {'--exp_name': 'ppo',
         '--env_name': 'PongNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '1'},
        {'--exp_name': 'ppo',
         '--env_name': 'FreewayNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '2'},
        {'--exp_name': 'ppo',
         '--env_name': 'BreakoutNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '3'},
        {'--exp_name': 'ppo',
         '--env_name': 'SpaceInvadersNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '1'},
        {'--exp_name': 'ppo',
         '--env_name': 'BowlingNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '2'},
        {'--exp_name': 'ppo',
         '--env_name': 'BoxingNoFrameskip-v4',
         '--param_name': 'baseline',
         '--gpu_device': '3'},
    ]
    def run_experiment(experiment):
        cmd = ['python', 'train_atari.py']
        print(experiment)
        for key, value in experiment.items():
            cmd.append(key)
            cmd.append(value)
        return subprocess.call(cmd)

    pool = Pool(6)
    pool.map(run_experiment, experiments)
    pool.close()
