# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of ProZero for dual inhibitor protein mutation

@author: dong
"""
from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sequence_env_m_p import Seq_env, Mutate
from mcts_alphaZero_mutate_gcgr_glp1r import MCTSMutater
from plm_v_net import PolicyValueNet
import sys
import datetime
import argparse
from vocab import AAS
import torch


class TrainPipeline():
    def __init__(self, starts, alphabet, model, trust_radius, init_model=None, cpuct=10):  # init_model=None
        start_seq = starts['start_peptide']
        self.seq_len = len(start_seq)
        self.vocab_size = len(alphabet)
        self.n_in_row = 4
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            starts,
            trust_radius)  # n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move 400   1600
        self.c_puct = cpuct  # 0.5  # 10
        self.buffer_size = 10000
        self.batch_size = 32  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 15000  # default is 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        # self_added
        self.buffer_no_extend = False
        # self_added
        # playout
        self.generated_seqs = []
        self.fit_list = []
        self.p_dict = {}
        self.m_p_dict = {}
        self.retrain_flag = False
        self.part = 2
        # playout
        #
        if init_model is not None:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len, self.vocab_size, model_file=init_model)
        else:  # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len, self.vocab_size)
        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        counts = len(self.generated_seqs)
        self.buffer_no_extend = False
        for _ in range(n_games):
            play_data, seq_and_fit, p_dict = self.mutate.start_mutating(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.p_dict = p_dict
            self.m_p_dict.update(self.p_dict)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                print('buffer size: ', len(self.data_buffer))
                for seq, fit in seq_and_fit:  # alphafold_d
                    if seq not in self.generated_seqs:
                        self.generated_seqs.append(seq)
                        self.fit_list.append(fit)
                        if seq not in self.m_p_dict.keys():
                            self.m_p_dict[seq] = fit
                        if len(self.generated_seqs) % 10 == 0 and len(self.generated_seqs) > counts and self.part <= 10:
                            self.retrain_flag = True

    def value_update(self):
        """update the value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        # mcts_probs_batch = [data[1] for data in mini_batch]
        reward_batch = [data[1] for data in mini_batch]
        old_v = self.policy_value_net.value(state_batch)
        for _ in range(self.epochs):
            loss = self.policy_value_net.train_step(
                state_batch,
                # mcts_probs_batch,
                reward_batch,
                self.learn_rate)
            new_v = self.policy_value_net.value(state_batch)

        explained_var_old = (1 - np.var(np.array(reward_batch).flatten() - old_v.flatten()) / (np.var(np.array(reward_batch).flatten()) + 1E-5))
        explained_var_new = (1 - np.var(np.array(reward_batch).flatten() - new_v.flatten()) / (np.var(np.array(reward_batch).flatten()) + 1E-5))
        print("training value loss:{}, explained_var_old:{:.3f}, explained_var_new:{:.3f}".format(loss, explained_var_old, explained_var_new))
        return loss

    def run(self):
        """run the training pipeline"""
        starttime = datetime.datetime.now()
        try:
            for i in range(self.game_batch_num):
                print('********************************game {} start****************************************'.format(i))
                self.collect_selfplay_data(self.play_batch_size)  # play_batch_size is 1
                print("episode i:{}, episode collects {} sequences".format(i + 1, self.episode_len))
                print('********************************game {} end******************************************'.format(i))
                if i == self.game_batch_num - 1 or len(self.m_p_dict.keys()) >= 4000:
                    m_p_fitness = np.array(list(self.m_p_dict.values()))
                    m_p_seqs = np.array(list(self.m_p_dict.keys()))
                    df_m_p = pd.DataFrame({"sequence": m_p_seqs, "GCGR": np.power(10, -(m_p_fitness[:, 0] + 10)) * 1E12, "GLP-1R": np.power(10, -(m_p_fitness[:, 1] + 10)) * 1E12})
                    df_m_p.to_csv("./results/GCGR_GLP-1R_generated_sequence_{}_test_{}_c_puct{}.csv".format(args.method, args.idx, args.cpuct), index=False)
                    endtime = datetime.datetime.now()
                    print('time cost：', (endtime - starttime).seconds)
                    sys.exit(0)
                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend is False:
                    self.value_update()
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    parser = argparse.ArgumentParser(description='ProZero_GCGR_GLP-1R')
    parser.add_argument("--idx", type=str, default="1", help="Test index")
    parser.add_argument("--method", type=str, default="prozero", help="method name")
    parser.add_argument("--gpus", type=str, default="0", help="GPU index")
    parser.add_argument("--cpuct", type=int, default=10, help="PUCT tradeoff factor")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device(int(args.gpus))
    print('GPUs are {}'.format(args.gpus))

    start_time = datetime.datetime.now()
    starts = {"start_peptide": "HSQGTFTSDYSKYLDSRAASEFVQWLISE"}  # with both strong potency, 2.08 and 1.59
    training_pipeline = TrainPipeline(starts, AAS, model=None, trust_radius=100, cpuct=args.cpuct)
    training_pipeline.run()
