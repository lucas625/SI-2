"""
Module for evaluating the learning.
"""

import csv
import os
import uuid


class Evaluator:
    """
    Class for evaluating the learning.
    """

    @staticmethod
    def evaluate(episodes_rewards, number_of_evaluation_intervals, csv_file_path):
        """
        Performs the evaluation of the rewards, writing to a csv.
        :param list[float] episodes_rewards: the list of the reward of each episode.
        :param int number_of_evaluation_intervals: the number of intervals between each evaluation.
        :param str csv_file_path: the path to write the csv file.
        """
        evaluation_interval = int(len(episodes_rewards) / number_of_evaluation_intervals)
        evaluations = list()
        evaluations.append(dict(interval=0, average_reward=0))
        for index in range(0, len(episodes_rewards), evaluation_interval):
            interval_end = index+evaluation_interval
            average_interval_reward = sum(episodes_rewards[index:interval_end]) / evaluation_interval
            evaluations.append(dict(interval=interval_end, average_reward=average_interval_reward))

        result_path = Evaluator._write_csv(evaluations, csv_file_path)
        print('Evaluation written to {}.'.format(result_path))

    @staticmethod
    def _write_csv(evaluations, csv_file_path):
        """
        Function for writing a csv file.
        :param list[dict] evaluations: the list of evaluations.
        :return str:
        """
        csv_path = Evaluator._generate_csv_path(csv_file_path)

        with open(csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['interval', 'average_reward'])
            writer.writeheader()
            writer.writerows([evaluation for evaluation in evaluations])
        return csv_path

    @staticmethod
    def _generate_csv_path(csv_file_path):
        """
        Generates the path to the csv file.
        :param str csv_file_path: the path to the csv file.
        :return str:
        """
        base_path = os.path.join(csv_file_path, 'evaluations')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        return os.path.join(base_path, '{}.csv'.format(uuid.uuid4()))
