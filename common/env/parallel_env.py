import copy
import numpy as np
from multiprocessing import Process, Pipe, Value


def worker(worker_id, env, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'seed':
            worker_end.send(env.seed(data))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError


class ParallelEnv(object):
    """
    This class
    """

    def __init__(self, num_processes, env):
        self.nenvs = num_processes
        self.waiting = False
        self.closed = False
        self.workers = []
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.master_ends, self.send_ends = zip(*[Pipe() for _ in range(self.nenvs)])
        for worker_id, (master_end, send_end) in enumerate(zip(self.master_ends, self.send_ends)):
            p = Process(target=worker,
                        args=(worker_id, copy.deepcopy(env), master_end, send_end))
            p.start()
            self.workers.append(p)

    def step(self, actions):
        """
        Perform step for each environment and return the stacked transitions
        """
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, master_end in enumerate(self.master_ends):
            master_end.send(('seed', seed + idx))
        return [master_end.recv() for master_end in self.master_ends]

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        results = [master_end.recv() for master_end in self.master_ends]

        return np.stack(results)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True