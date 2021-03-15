import numpy as np
import unittest
import copy
from classifier import XCSF_Classifier
from matching import XCSFMatching
from action_selection import XCSFActionSelection
from reinforcement import Reinforcement
from genetic_algorithm import CIGeneticAlgorithm
import os
import random
import pickle
from xcsf_er import ExperienceReplay


'''
class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, experience):
        self.memory.append(experience)

    def get_batch(self, batch_size=10):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory

        return batch
'''


class XCSF_ER_transfer:

    GAMMA = 0.71

    def transfer_knowledge(self, file_path):
        with open(file_path, 'rb') as pickle_file:
            old_lcs = pickle.load(pickle_file)
        self.population = old_lcs.population
        self.max_population_size = old_lcs.max_population_size
        self.histlen = old_lcs.histlen
        # only take above average rules --> experimental
        # avg_fitness = sum(list(map(lambda x: x.fitness, self.population))) / len(self.population)
        # self.population = list(filter(lambda x: x.fitness >= avg_fitness, self.population))
        # set to defaults
        for classifier in self.population:
            classifier.reset()

    def retrieve_filename(self, reward_func, dataset, save_dest):
        file_name = save_dest.split(os.sep)[-1]
        file_name = file_name.replace(self.name, "XCSF_ER")
        if not self.is_histlen_exp:
            prefix = "rq_" + "XCSF_ER"
            intended_data_set = file_name[len(prefix) + 1:].split("_")[0]
            file_name = file_name.replace(intended_data_set, dataset)
            prefix += "_" + dataset
            intended_reward_func = file_name[len(prefix) + 1:].split("_")[0]
            file_name = file_name.replace(intended_reward_func, reward_func)
        else:
            pass
        return "OLD_AGENTS" + os.sep + file_name + ".p"

    def initialize_transfer_learning(self, save_dataset):
        load_from = self.retrieve_filename(self.reward_func_knowledge_base, self.knowledge_base, save_dataset)
        self.transfer_knowledge(load_from)

    def __init__(self, reward_func, dataset, possible_actions=None, is_histlen_exp=False):
        '''
        :param from_data_set: which dataset to use
        '''

        print("init XCSF_ER_transfer")
        self.knowledge_base = dataset
        self.reward_func_knowledge_base = reward_func
        self.is_histlen_exp = is_histlen_exp
        self.action_size = 42
        self.name = "XCSF_ER_transfer" + "_from_" + dataset + "_" + reward_func + "_reward"
        if possible_actions is None:
            self.possible_actions = [-10, 10]
        else:
            self.possible_actions = possible_actions
        self.time_stamp = 1
        self.action_history = []
        self.old_action_history = []
        self.reinforce = Reinforcement()
        self.ga = CIGeneticAlgorithm(possible_actions)
        #################################
        self.single_testcases = True
        #################################
        # stuff for batch update
        self.max_prediction_sum = 0
        self.rewards = None
        self.p_explore = 0.25
        self.train_mode = True
        self.cycle = 0
        # stuf for er
        self.batch_size = 2000
        self.buffer = ExperienceReplay(12000)

    def get_action(self, state):
        '''
        :param state: State in Retects. In the XCS world = situation.

        :return : a action
        '''
        theta_mna = 45 # len(self.possible_actions)
        matcher = XCSFMatching(theta_mna, self.possible_actions)
        match_set = matcher.get_match_set(self.population, state, self.time_stamp)
        self.p_explore = (self.p_explore - 0.1) * 0.99 + 0.1
        action_selector = XCSFActionSelection(self.possible_actions, self.p_explore)
        best_possible_action = action_selector.get_best_action(match_set, state)
        chosen_action = action_selector.get_action(best_possible_action, self.train_mode)
        # calculate system prediction
        fitness_sum = sum(list(map(lambda x: x.fitness, self.population)))
        system_prediction = sum(list(map(lambda x: x.fitness * x.get_target(state, chosen_action), self.population)))
        if fitness_sum > 0:
            system_prediction = system_prediction / fitness_sum
        max_val = system_prediction # on policy
        action_set = match_set
        # action_set = action_selector.get_action_set(match_set, action)
        self.max_prediction_sum += max_val
        self.action_history.append((state, action_set, chosen_action))
        return system_prediction
        # return chosen_action

    def reward(self, new_rewards):
        try:
            x = float(new_rewards)
            new_rewards = [x] * len(self.action_history)
        except Exception as _:
            if len(new_rewards) < len(self.action_history):
                raise Exception('Too few rewards')
        # old_rewards = self.rewards
        self.rewards = new_rewards
        old_rewards = self.rewards
        if old_rewards is not None:
            # avg_max_pred = self.max_prediction_sum / len(self.action_history)
            for i in range(0, len(old_rewards)):
                discounted_reward = old_rewards[i] #+ XCSF.GAMMA * avg_max_pred
                old_sigma, _, old_action = self.action_history[i] # self.old_action_history[i]
                self.buffer.remember((old_sigma, old_action, discounted_reward))
        if self.cycle % 3 == 0 or self.cycle == 2:
            self.learn_from_experience()
        self.max_prediction_sum = 0
        self.old_action_history = self.action_history
        self.action_history = []
        self.delete_from_population()
        print(self.name + " : finished cycle " + str(self.cycle))
        self.cycle += 1

    def learn_from_experience(self):
        batch = self.buffer.get_batch(self.batch_size)
        for experience in batch:
            state, action, reward = experience
            theta_mna = 45  # len(self.possible_actions)
            matcher = XCSFMatching(theta_mna, self.possible_actions)
            match_set = matcher.get_match_set(self.population, state, self.time_stamp)
            self.reinforce.reinforce_xcsf(match_set, reward, state, 0)
            self.ga.perform_iteration(match_set, state, self.population, self.time_stamp)
            self.time_stamp += 1

    def delete_from_population(self):
        '''
        Deletes as many classifiers as necessary until the population size is within the
        defined bounds.
        '''
        total_numerosity = sum(list(map(lambda x: x.numerosity, self.population)))
        while len(self.population) > self.max_population_size:
            total_fitness = sum(list(map(lambda x: x.fitness, self.population)))
            avg_fitness = total_fitness / total_numerosity
            vote_sum = sum(list(map(lambda x: x.deletion_vote(avg_fitness), self.population)))
            choice_point = random.random() * vote_sum
            vote_sum = 0
            for classifier in self.population:
                vote_sum += classifier.deletion_vote(avg_fitness)
                if vote_sum > choice_point:
                    if classifier.numerosity > 1:
                        classifier.numerosity = classifier.numerosity - 1
                    else:
                        self.population.remove(classifier)

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


if __name__ == "__main__":
    dataset = "gsdtsr"
    reward = "timerank"
    save_dest = "fu" + os.sep + "rq_XCSF_ER_gsdtsr_timerank_16_agent.p"
    agent = XCSF_ER_transfer(reward, dataset)
    agent.initialize_transfer_learning(save_dest)
    print('%s_agent' % "file")
    print("hey ha")
