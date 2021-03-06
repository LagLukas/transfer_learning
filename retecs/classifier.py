from abc import abstractmethod
import sys
import unittest
import random


class Classifier:

    def __init__(self):
        pass

    @abstractmethod
    def matches(self, sigma):
        '''
        :param sigma: situation.
        '''
        pass


class CIClassifier(Classifier):

    THETA_DEL = 20
    DELTA = 0.1
    HISTORY_LENGTH = 1
    EPSILON_I = 0
    P_I = 0
    F_I = 0

    def __init__(self, previous_results, last_execution, average_duration, action, timestamp):
        '''
        :param previous_results: test result conditions as ternary state.
        :param last_execution: tuple for the timestamp conditions.
        :param average_duration: time estimation of the test to run as a intervall.
        :param action: pruposed action
        :param timestamp: time of generation
        '''
        self.previous_results_conditions = previous_results
        self.last_executions_condition = last_execution
        self.average_duration_condition = average_duration
        self.action = action
        self.timestamp = timestamp
        self.fitness = CIClassifier.F_I
        self.prediction = CIClassifier.P_I
        self.epsilon = CIClassifier.EPSILON_I
        self.experience = 0
        self.action_set_size = 1
        self.numerosity = 1

    def matches(self, sigma):
        '''
        :param sigma: dictionary with the keys previous_results (list of booleans),
        last_executions (list of non negative integers), duration (float).

        :return : boolean indicating the situation sigma fullfills the conditions of the classifier.
        '''
        duration = sigma["duration"]
        condition = self.average_duration_condition
        if not (duration >= condition[0] and duration <= condition[1]):
            return False
        for i in range(0, len(sigma["previous_results"])):
            result_i = sigma["previous_results"][i]
            result_condition_i = self.previous_results_conditions[i]
            if result_condition_i != "#" and result_condition_i != result_i:
                return False
        execution = sigma["last_execution"]
        execution_condition = self.last_executions_condition
        if not (execution_condition[0] <= execution and execution <= execution_condition[1]):
            return False
        return True

    def deletion_vote(self, avg_fitness):
        '''
        Calculates the deletion vote of the classifier based on the average fitness of the population,
        the classifiers fitness and its experience.

        :param avg_fitness: average fitness of the entire population.

        :return : deletion vote of the classifier
        '''
        vote = self.action_set_size * self.numerosity
        if self.experience > CIClassifier.THETA_DEL and (self.fitness / self.numerosity) < CIClassifier.DELTA * avg_fitness:
            if self.fitness > 0:
                vote = (vote * avg_fitness) / (self.fitness / self.numerosity)
            else:
                vote = sys.maxsize
        return vote


class XCSF_Classifier(CIClassifier):

    WEIGHTS_LOWER = -1.0
    WEIGHTS_UPPER = 1.0

    @staticmethod
    def get_rand_weight():
        r = random.random() * (XCSF_Classifier.WEIGHTS_UPPER - XCSF_Classifier.WEIGHTS_LOWER)
        return r + XCSF_Classifier.WEIGHTS_LOWER

    def get_target(self, sigma, action):
        target = 0 # self.action_weights[0] * action + self.action_weights[1] * (action ** 2)
        target += self.state_weights[0] * sigma["duration"]
        target += self.state_weights[1] * sigma["last_execution"]
        for i in range(0, len(sigma["previous_results"])):
            curr = 1 if sigma["previous_results"][i] is True else 0
            target += self.state_weights[2 + i] * curr
        target += self.intercept_weight
        return target

    def update_weights(self, update_factor, sigma, action):
        # use modified delta rule
        # self.action_weights[0] += update_factor * action
        # self.action_weights[1] += update_factor * (action ** 2)
        self.state_weights[0] += update_factor * sigma["duration"]
        self.state_weights[1] += update_factor * sigma["last_execution"]
        for i in range(0, len(sigma["previous_results"])):
            curr = 1 if sigma["previous_results"][i] is True else 0
            self.state_weights[2 + i] += update_factor * curr
        self.intercept_weight += update_factor * 1

    def __init__(self, previous_results, last_execution, average_duration, action, timestamp, state_weights, action_weights, intercept_weight):
        super().__init__(previous_results, last_execution, average_duration, action, timestamp)
        self.state_weights = state_weights
        self.action_weights = action_weights
        self.intercept_weight = intercept_weight
        self.action = "dummy"

    def reset(self):
        # self.fitness = 0.1 * self.fitness
        self.fitness = CIClassifier.F_I
        self.prediction = CIClassifier.P_I
        self.epsilon = CIClassifier.EPSILON_I
        self.experience = 0
        self.numerosity = 1

    def convert_python_to_json(self):
        data_dict = {}
        data_dict["previous_results_conditions"] = self.previous_results_conditions
        data_dict["last_executions_condition"] = self.last_executions_condition
        data_dict["average_duration_condition"] = self.average_duration_condition
        data_dict["action"] = self.action
        data_dict["timestamp"] = self.timestamp
        data_dict["fitness"] = self.fitness
        data_dict["prediction"] = self.prediction
        data_dict["epsilon"] = self.epsilon
        data_dict["experience"] = self.experience
        data_dict["action_set_size"] = self.action_set_size
        data_dict["numerosity"] = self.numerosity
        # XCSF stuff
        data_dict["state_weights"] = self.state_weights
        data_dict["action_weights"] = self.action_weights
        data_dict["intercept_weight"] = self.intercept_weight
        return data_dict


class Test_CIClassifier(unittest.TestCase):

    def test_deletion_vote(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.fitness = 10.0
        classifier.action_set_size = 10.0
        classifier.experience = 100.0
        classifier.numerosity = 2.0
        vote = classifier.deletion_vote(100)
        assert vote == 400.0

    def test_match(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        situation = {}
        situation["duration"] = 43
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        matches = classifier.matches(situation)
        assert matches == True

    def test_no_match(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        situation = {}
        situation["duration"] = 47
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        matches = classifier.matches(situation)
        assert matches == False
        situation["duration"] = 44
        situation["previous_results"] = [True, True, True]
        situation["last_execution"] = 2
        matches = classifier.matches(situation)
        assert matches == False
        situation["duration"] = 44
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 42
        matches = classifier.matches(situation)
        assert matches == False

if "__main__" == __name__:
    unittest.main()

