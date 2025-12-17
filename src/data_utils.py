#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import csv
import os 
from datetime import datetime


def save_myrewire_algorithm(algorithm, test_acc, best_val_acc, Gamma_0, parameters):
	headers = ['algorithm', 'current_time', 'test_acc', 'best_val_acc', 'Gamma_0', 'budget_delete',
				'budget_add', 'num_additions', 'before_edge_cnt', 'after_edge_cnt']
	current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	filename = "result_cbrr.csv"
	parameters_str = json.dumps(parameters)

	budget_delete = parameters.get('budget_delete', 0)
	budget_add = parameters.get('budget_add', 0)
	num_additions = parameters.get('num_additions', 0)
	before_edge_cnt = parameters.get('before_edge_cnt', 0)
	after_edge_cnt = parameters.get('after_edge_cnt', 0)

	record_line = [
		algorithm,
		str(current_time),
		str(test_acc),
		str(best_val_acc),
		str(Gamma_0),
		str(budget_delete),
		str(budget_add),
		str(num_additions),
		str(before_edge_cnt),
		str(after_edge_cnt)
	] 
	with open(filename, mode='a', newline='') as file:
		writer = csv.writer(file)
		if file.tell() == 0:
			writer.writerow(headers)
		writer.writerow(record_line)


def save_result_1(algorithm, parameters, test_acc, best_val_acc, Gamma_0):
	if algorithm == 'cbrr':
		save_myrewire_algorithm(algorithm, test_acc, best_val_acc, Gamma_0, parameters)
		return

	headers = ['algorithm', 'parameters', 'test_acc', 'best_val_acc', 'Gamma_0', 'current_time']
	current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	filename = "result.csv"
	parameters_str = json.dumps(parameters)

	record_line = [
		algorithm, 
		parameters_str,
		str(test_acc),
		str(best_val_acc),
		str(Gamma_0),
		str(current_time)
	]

	with open(filename, mode='a', newline='') as file:
		writer = csv.writer(file)
		if file.tell() == 0:
			writer.writerow(headers)
		writer.writerow(record_line)

