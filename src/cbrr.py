#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import networkx as nx
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import itertools


def convert_to_graph(data, is_undirected=True):
	N = data.x.shape[0]
	M = data.edge_index.shape[1]

	if not 'edge_type' in data.keys():
		edge_type = np.zeros(M, dtype=int)
	else:
		edge_type = data.edge_type
	g = to_networkx(data)
	if is_undirected:
		g = g.to_undirected()
	return g, N, edge_type 


def _compute_effective_resistance(L, num_nodes, approximate):
	rcond = 1e-5 if approximate else 1e-20
	L_inv = np.linalg.pinv(L, rcond=rcond)
	L_inv_diag = np.diag(L_inv)
	L_inv_aa = np.broadcast_to(L_inv_diag, (num_nodes, num_nodes))
	R = L_inv_aa - 2 * L_inv + L_inv_aa.T 
	return R


def compute_effective_resistance(graph, approximate=False):
	num_nodes = graph.number_of_nodes()
	L = nx.laplacian_matrix(graph, list(graph)).toarray().astype(np.float64)
	R = _compute_effective_resistance(L, num_nodes, approximate)
	return R


def community_effective_resistance(graph, communities, num_edges):
	sub_graph = graph.subgraph(list(set(communities)))
	node_list = list(sub_graph.nodes)
	added_edges = []
	R = None
	for _ in range(num_edges):
		try:
			new_R = compute_effective_resistance(sub_graph, approximate=False) 
			R = new_R
		except:
			pass
		if R is None:
			continue
		max_resistance_index = np.unravel_index(R.argmax(), R.shape)
		added_edges.append(
			(node_list[max_resistance_index[0]],
			 node_list[max_resistance_index[1]])
		)
		R[max_resistance_index] = 0
	return added_edges


def edge_rewire(data, budget_add, budget_delete, num_additions, community_resolution, init_seed_value=0):
	# step 1
	g, N, edge_type = convert_to_graph(data, is_undirected=True) 
	communities = list(nx.community.louvain_communities(g, resolution=community_resolution, seed=init_seed_value)) 

	cluster_dict_before = {node: i for i, cluster in enumerate(communities) for node in cluster}
	cluster_list_before = [cluster_dict_before[node] for node in range(len(data.y))]
	nmiscoremod_before = NMI(cluster_list_before, data.y.cpu().numpy())
	original_edge_count = g.number_of_edges()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	features = data.x.to(device)

	similarities = torch.mm(features, features.t())
	norms = torch.norm(features, dim=1, keepdim=True)
	similarities = similarities / (norms * norms.t())
	similarities = similarities.cpu().numpy()

	if np.isnan(similarities).any():
		print("NaN detected in similarities matrix")

	# step 2
	new_communities = []
	for i, comm1 in enumerate(communities):
		new_edge_index = community_effective_resistance(g, comm1, num_additions)
		for u,v in new_edge_index:
			g.add_edge(u, v)

	# step 3 
	scores = { (i, j): (len(comm1) * len(comm2)) 
				for i, comm1 in enumerate(communities) for j, comm2 in enumerate(communities) if i <= j }
	norm_scores = { (i, j): scores[(i, j)] / sum(scores.values()) for i, j in scores }   # normalization 

	budgets_add = { (i, j): int(budget_add * norm_scores[(i, j)] ) for i, j in norm_scores }
	budgets_delete = { (i, j): int(budget_delete * norm_scores[(i, j)] ) for i, j in norm_scores }

	edges_added = set()
	edges_removed = set()

	total_comparisons = sum(1 for i in range(len(communities)) for j in range(i, len(communities)))
	with tqdm(total=total_comparisons, desc="Processing communities") as pbar:
		for i, comm1 in enumerate(communities):
			for j, comm2 in enumerate(communities[i:], start=i):
				if i > j:
					continue  # Skip redundant comparisons
				comm1_set = set(comm1)
				comm2_set = set(comm2)

				edges_between = set(g.edges(comm1_set)) & set((u, v) for u in comm1_set for v in comm2_set) 
				if not edges_between:
					pbar.update(1)
					continue
				comm1_arr = np.array(list(comm1_set))
				comm2_arr = np.array(list(comm2_set)) 
				sim_matrix = similarities[comm1_arr[:, None], comm2_arr]

				edge_indices = np.array([(comm1_arr.tolist().index(u), comm2_arr.tolist().index(v)) for u, v in edges_between]) 
				sim_values = sim_matrix[edge_indices[:, 0], edge_indices[:, 1]]

				sim = np.mean(sim_values) if sim_values.size > 0 else 0 
				num_edges = len(edges_between)   # 计算iter-community的边的数量 

				ranking_remove = [(u, v, (sim*num_edges - similarities[u, v])/(num_edges - 1)) for u, v in edges_between if similarities[u, v] < sim] 
				non_edges = set(itertools.product(comm1_set, comm2_set)) - set(g.edges()) - {(v, u) for u, v in g.edges()} 
				ranking_add = [(u, v, (sim*num_edges + similarities[u, v])/(num_edges + 1)) for u, v in non_edges if similarities[u, v] > sim]  
				ranking_add_s = np.array(ranking_add, dtype=[('u', int), ('v', int), ('score', float)]) 
				ranking_remove_s = np.array(ranking_remove, dtype=[('u', int), ('v', int), ('score', float)]) 
				ranking_add_s.sort(order='score') 
				ranking_remove_s.sort(order='score')  

				edges_to_add = min(budgets_add[(i, j)], len(ranking_add_s)) 

				for u, v, _ in ranking_add_s[-edges_to_add:]:
					if (u, v) not in edges_added and (v, u) not in edges_added: 
						if len(edges_added) < budget_add:
							g.add_edge(u, v)
							edges_added.add((min(u, v), max(u, v)))  # Store only one direction
						else:
							break

				# Remove edges
				edges_to_remove = min(budgets_delete[(i, j)], len(ranking_remove_s)) 
				for u, v, _ in ranking_remove_s[-edges_to_remove:]:
					if (u, v) in g.edges() or (v, u) in g.edges():
						if len(edges_removed) < budget_delete:
							g.remove_edge(u, v)
							edges_removed.add((min(u, v), max(u, v)))  # Store only one direction
						else:
							break
				pbar.update(1)
	
	final_edge_count = g.number_of_edges()
	edges_modified = final_edge_count - original_edge_count 

	added_edges = len(edges_added)
	deleted_edges = len(edges_removed)
	total_edges = g.number_of_edges()

	num_edges_rewired_per_comm = {(i, j): 0 for i in range(len(communities)) for j in range(i, len(communities))}
	for u, v in edges_added.union(edges_removed):
		comm_i = [i for i, comm in enumerate(communities) if u in comm][0]
		comm_j = [i for i, comm in enumerate(communities) if v in comm][0]
		if comm_i < comm_j:
			num_edges_rewired_per_comm[(comm_i, comm_j)] += 1
		else:
			num_edges_rewired_per_comm[(comm_j, comm_i)] += 1
	num_edges_rewired_per_comm_normalized = {(i, j): num_edges_rewired_per_comm[(i, j)] / (len(communities[i]) * len(communities[j])) for i, j in num_edges_rewired_per_comm} 
	if np.isnan(list(num_edges_rewired_per_comm_normalized.values())).any():
		print("NaN detected in num_edges_rewired_per_comm_normalized")

	pyg_data = from_networkx(g)
	pyg_data.x = data.x
	pyg_data.y = data.y
	pyg_data.train_mask = data.train_mask
	pyg_data.val_mask = data.val_mask
	pyg_data.test_mask = data.test_mask 

	new_edge_index = pyg_data.edge_index 
	return new_edge_index, pyg_data

