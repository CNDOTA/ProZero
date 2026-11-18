# -*- coding: utf-8 -*-
from __future__ import print_function
import os.path
import time
import numpy as np
import torch
import random
from typing import List, Union
import copy
from bertrand.bertrand.model.inference import get_bertrand_prediction
from mhcflurry import Class1PresentationPredictor
mhcflurry_predictor = Class1PresentationPredictor.load()
from vocab import AAS, SCORE_DIM, SCORE_LIST
import subprocess


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


# 20240903: choose a better seed sequence by score sum
def mutate_sequence_ranking(peptide_sequence, ex_dict, episode_sequences):
    """Mutate the amino acid sequence by ranking
    """
    sorted_sequences = sorted(ex_dict, key=lambda k: sum(ex_dict[k]), reverse=True)  # decreasingly sorting by score sum
    for new_seq in sorted_sequences:
        if new_seq != peptide_sequence and new_seq not in episode_sequences:
            return new_seq
    raise ValueError('Not new sequence can be found.')

# def mutate_sequence(peptide_sequence, ex_list):
#     """Mutate the amino acid sequence randomly
#     """
#     restypes = list(AAS)
#     seqlen = len(peptide_sequence)
#     # mutate three residues until a new sequence is generated
#     while True:
#         seeds = peptide_sequence
#         pi_s = np.random.choice(np.arange(seqlen), size=2, replace=False)  # True,mutate<=3, False=3
#         for pi in pi_s:
#             aa = np.random.choice(restypes, replace=False)
#             # new_s
#             new_seq = seeds[:pi] + aa + seeds[pi + 1:]
#             seeds = new_seq
#         if new_seq not in ex_list:
#             break
#     return new_seq


def get_score_vector(bertrand_model, ini_pep, cur_pep, hla, cdr3b):
    score_vector = []
    if 'Bertrand' in SCORE_LIST:
        ptcr_proba = get_bertrand_prediction(model=bertrand_model, pep=cur_pep, cdr3b=cdr3b)
        score_vector.append(ptcr_proba)
    if 'NetMHCpan' in SCORE_LIST:
        rank_score = get_netmhcpan_prediction(cur_pep=cur_pep, hla=hla)
        score_vector.append(rank_score)
    if 'HHSuite' in SCORE_LIST:
        homo_proba = get_hhsuite_prediction(cur_pep=cur_pep, ini_pep=ini_pep)
        score_vector.append(homo_proba)
    if 'MHCflurry' in SCORE_LIST:
        pres_score = get_mhcflurry_prediction(cur_pep=cur_pep, hla=hla)
        score_vector.append(pres_score)
    if 'NetMHCstab' in SCORE_LIST:
        stab_score = get_netmhcstab_prediction(cur_pep=cur_pep, hla=hla)
        score_vector.append(stab_score)
    return score_vector


def get_netmhcpan_prediction(cur_pep, hla):
    test_pep_file = './netMHCpan-4.1/test/test_{}.pep'.format(str(time.time()) + str(random.random()))
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    with open(test_pep_file, 'w') as f:
        f.write(cur_pep)
    launch_string = './netMHCpan-4.1/netMHCpan -p {} -BA -a {} | grep {}'.format(test_pep_file, hla, cur_pep)
    p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
    p.wait()
    rank_score = p.communicate()[0].decode().split()[12]
    rank_score = 1 - float(rank_score) / 100  # scale rank_score to [0, 1] and the higher the better
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    return rank_score


def get_netmhcstab_prediction(cur_pep, hla):
    test_pep_file = './netMHCstabpan-1.0/test/test_{}.pep'.format(str(time.time()) + str(random.random()))
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    with open(test_pep_file, 'w') as f:
        f.write(cur_pep)
    launch_string = './netMHCstabpan-1.0/netMHCstabpan -p {} -a {} | grep {}'.format(test_pep_file, hla, cur_pep)
    p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
    p.wait()
    stab_score = float(p.communicate()[0].decode().split()[4])
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    return stab_score


def get_mhcflurry_prediction(cur_pep, hla):
    pres_score = mhcflurry_predictor.predict(peptides=[cur_pep], alleles=[hla], verbose=0)['presentation_score'][0]
    return pres_score


def get_hhsuite_prediction(cur_pep, ini_pep):
    # if the homology probability > 90%, then return 1.
    # should not consider the homology of the same peptide
    time_suffix = str(time.time()) + str(random.random())
    if cur_pep == ini_pep:
        homo_prob = 0
        return homo_prob
    test_pep_file = './hhsuite/test_{}.pep'.format(time_suffix)
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    with open(test_pep_file, 'w') as f:
        f.write('>' + cur_pep + '\n' + cur_pep)
    init_pep_file = './hhsuite/init_{}.pep'.format(time_suffix)
    if not os.path.exists(init_pep_file):
        with open(init_pep_file, 'w') as f:
            f.write('>' + ini_pep + '\n' + ini_pep)
    # hhalign -i input.fasta -t template.fasta
    launch_string = 'hhalign -i {} -t {} -v 0'.format(test_pep_file, init_pep_file, cur_pep)
    p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
    p.wait()
    p.communicate()
    result_file = './hhsuite/test_{}.hhr'.format(time_suffix)
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            # 9 is line number and 2 is index number
            homo_prob = float(f.readlines()[9].split()[2]) / 100
    if os.path.exists(test_pep_file):
        os.remove(test_pep_file)
    if os.path.exists(init_pep_file):
        os.remove(init_pep_file)
    if os.path.exists(result_file):
        os.remove(result_file)
    return homo_prob > 0.9


def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.
    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing a sequence.
        alphabet: Alphabet string (assigns each character an index).
    Returns:
        Sequence string representation of `one_hot`.
    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def string_to_feature(string):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots


class Seq_env(object):
    """sequence space for the env"""
    def __init__(self,
                 seq_len,
                 alphabet,
                 model,
                 starts,
                 trust_radus,
                 ):

        self.max_moves = trust_radus
        self.move_count = 0

        self.seq_len = seq_len  # self.width = int(kwargs.get('width', 8))
        self.vocab_size = len(alphabet)  # self.height = int(kwargs.get('height', 8))

        self.alphabet = alphabet
        self.model = model
        starting_seq = starts['start_peptide']
        self.cdr3b = starts['cdr3b']
        self.hla = starts['hla'].replace('*', '')
        self.starting_seq = starting_seq
        self.seq = starting_seq

        self._state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)

        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.unuseful_move = 0
        self.states = {}
        self.episode_seqs = []
        self.episode_seqs.append(starting_seq)
        self.repeated_seq_ocurr = False
        self.init_state_count = 0
        # playout
        self.start_seq_exclude_list = []
        self.playout_dict = {}
        self.model.eval()

    def init_seq_state(self):  # start_player=0
        self.previous_fitness = [-float("inf")] * SCORE_DIM
        self.move_count = 0
        self.unuseful_move = 0
        self.repeated_seq_ocurr = False

        self._state = copy.deepcopy(self.init_state)
        combo = one_hot_to_string(self._state, AAS)
        self.start_seq_exclude_list.append(combo)
        self.init_combo = combo

        if combo not in self.episode_seqs:
            self.episode_seqs.append(combo)

        with torch.no_grad():
            outputs = get_score_vector(self.model, self.starting_seq, cur_pep=combo, hla=self.hla, cdr3b=self.cdr3b)

        if outputs:
            self._state_fitness = outputs

        self.availables = list(range(self.seq_len * self.vocab_size))
        # evo
        for i, a in enumerate(combo):
            self.availables.remove(self.vocab_size * i + AAS.index(a))
        for i, e_s in enumerate(self.episode_seqs):
            a_e_s = string_to_one_hot(e_s, AAS)
            a_e_s_ex = np.expand_dims(a_e_s, axis=0)
            if i == 0:
                nda = a_e_s_ex
            else:
                nda = np.concatenate((nda, a_e_s_ex), axis=0)

        c_i_s = string_to_one_hot(combo, AAS)
        for i, aa in enumerate(combo):
            tmp_c_i_s = np.delete(c_i_s, i, axis=0)
            for slice in nda:
                tmp_slice = np.delete(slice, i, axis=0)
                if (tmp_c_i_s == tmp_slice).all():
                    bias = np.where(slice[i] != 0)[0][0]
                    to_be_removed = self.vocab_size * i + bias
                    if to_be_removed in self.availables:
                        self.availables.remove(to_be_removed)
        # evo
        self.states = {}
        self.last_move = -1
        self.previous_init_state = copy.deepcopy(self._state)

    def current_state(self):
        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T

    def do_mutate(self, move, playout=True):
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = [0.0] * SCORE_DIM  # two reward functions [phla_proba, rank_score]
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1
            combo = one_hot_to_string(self._state, AAS)
            if not playout:
                if combo not in self.playout_dict.keys():
                    with torch.no_grad():
                        outputs = get_score_vector(self.model, self.starting_seq, cur_pep=combo, hla=self.hla, cdr3b=self.cdr3b)
                    if outputs:
                        self._state_fitness = outputs
                        # self.playout_dict[combo] = outputs  # if not rollout, do not add it into the rollout dict
                else:
                    self._state_fitness = self.playout_dict[combo]
            else:  # in the rollout process
                if combo not in self.playout_dict.keys():
                    with torch.no_grad():
                        outputs = get_score_vector(self.model, self.starting_seq, cur_pep=combo, hla=self.hla, cdr3b=self.cdr3b)
                    if outputs:
                        self._state_fitness = outputs
                        self.playout_dict[combo] = outputs
                else:
                    self._state_fitness = self.playout_dict[combo]

        current_seq = one_hot_to_string(self._state, AAS)
        # generate the same peptide, 20240619: check whether this operation will affect the performance
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            # 20240619: the visit time will control the rollout, do not change the original rewards of node
            # self._state_fitness = [0.0, 0.0]  # two reward functions
        else:
            self.episode_seqs.append(current_seq)
        # generate peptide with higher rewards, # and not repeated_seq_ocurr:  # 0.6* 0.75*
        if (np.asarray(self._state_fitness) >= np.asarray(self.previous_fitness)).all():
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0
        self.last_move = move

    def mutation_end(self):
        if self.repeated_seq_ocurr:
            return True
        if self.move_count >= self.max_moves:
            return True
        if self.unuseful_move == 1:
            return True
        if (np.asarray(self._state_fitness) < np.asarray(self.previous_fitness)).any():  # 0.6* 0.75*
            return True
        return False


class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env

    def start_mutating(self, mutater, is_shown=0, temp=1e-3, jumpout=25):
        """ start mutating using an MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        if (self.Seq_env.previous_init_state == self.Seq_env.init_state).all():
            self.Seq_env.init_state_count += 1
        if self.Seq_env.init_state_count >= jumpout:
            print("Random start replacement****")
            # current_start_seq = one_hot_to_string(self.Seq_env.init_state, AAS)
            # 20240621: mutate from the starting sequence to keep high homology
            current_start_seq = self.Seq_env.starting_seq
            episode_seqs = copy.deepcopy(self.Seq_env.episode_seqs)
            # playout_seqs = copy.deepcopy(list(self.Seq_env.playout_dict.keys()))
            # e_p_list = list(set(episode_seqs + playout_seqs))
            # new_start_seq = mutate_sequence(current_start_seq, e_p_list)
            new_start_seq = mutate_sequence_ranking(current_start_seq, self.Seq_env.playout_dict, episode_seqs)
            self.Seq_env.init_state = string_to_one_hot(new_start_seq, self.Seq_env.alphabet).astype(np.float32)
            self.Seq_env.init_state_count = 0

        self.Seq_env.init_seq_state()
        if self.Seq_env.init_combo in self.Seq_env.playout_dict.keys():
            print("starting sequence：{} with repeat time: {} and metric {}".format(self.Seq_env.init_combo, self.Seq_env.init_state_count, self.Seq_env.playout_dict[self.Seq_env.init_combo]))
        else:
            print("starting sequence：{} with repeat time: {}".format(self.Seq_env.init_combo, self.Seq_env.init_state_count))
        generated_seqs = []

        fit_result = []
        states, reward_z = [], []  # current_players
        game_step = 0
        while True:
            move, play_seqs, play_losses, play_to_buffer_dict = mutater.get_action(self.Seq_env, temp=temp, return_prob=False, game_step=game_step)
            # print('collect ', len(play_to_buffer_dict))
            self.Seq_env.playout_dict.update(mutater.m_p_dict)
            if move is not None:
                # store the data
                # states.append(self.Seq_env.current_state())
                # mcts_probs.append(move_probs)
                # reward_z.append(self.Seq_env._state_fitness)
                for seq in play_to_buffer_dict:
                    states.append(play_to_buffer_dict[seq][0])
                    reward_z.append(play_to_buffer_dict[seq][1])
                # perform a move
                self.Seq_env.do_mutate(move)
                generated_seqs.append(one_hot_to_string(self.Seq_env._state, AAS))

                fit_result.append(self.Seq_env._state_fitness)
                print('game step {} with action {}'.format(game_step, move))
                print("move_fitness: {}".format(self.Seq_env._state_fitness))
                print("episode_seq len: %d" % (len(self.Seq_env.episode_seqs)))
                print("Mmove & playout dict len: %d" % (len(self.Seq_env.playout_dict)))
                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print(state_string)
                game_step += 1
            end = self.Seq_env.mutation_end()
            if end:
                mutater.reset_Mutater()
                print("Mutation end.")
                if is_shown:
                    print("Mutation end.")
                playout_dict = copy.deepcopy(self.Seq_env.playout_dict)
                return zip(states, reward_z), zip(generated_seqs, fit_result), playout_dict
