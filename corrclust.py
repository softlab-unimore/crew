import numpy as np

from pyccalg import kwikcluster, _check_clustering, _map_cluster, _vertex_pair_ids, _linear_program_scipy, \
    round_charikar, round_demaine, _solve_lp_scipy, _lp_solution_cost

random_edgeweight_generation = None
edge_addition_prob = -1
solver = 'scipy'
# algorithm = 'demaine'
eps = 0.0000000001


def _load(dataset: list[list]):
    tot_min = 0
    id2vertex = {}
    vertex2id = {}
    edges = []
    graph = {}
    vertex_id = 0
    for tokens in dataset:
        u = int(tokens[0])
        v = int(tokens[1])
        wp = float(tokens[2])
        wn = float(tokens[3])
        if wp != wn:
            if u not in vertex2id:
                vertex2id[u] = vertex_id
                id2vertex[vertex_id] = u
                vertex_id += 1
            if v not in vertex2id:
                vertex2id[v] = vertex_id
                id2vertex[vertex_id] = v
                vertex_id += 1
            uid = vertex2id[u]
            vid = vertex2id[v]
            if uid < vid:
                edges.append((uid, vid))
            else:
                edges.append((vid, uid))
            if uid not in graph.keys():
                graph[uid] = {}
            if vid not in graph.keys():
                graph[vid] = {}
            min_pn = min(wp, wn)
            tot_min += min_pn
            graph[uid][vid] = (wp - min_pn, wn - min_pn)
            graph[vid][uid] = (wp - min_pn, wn - min_pn)

    return id2vertex, vertex2id, edges, graph, tot_min


def corrclust(dataset, algorithm):
    (id2vertex, vertex2id, edges, graph, tot_min) = _load(dataset)
    n = len(id2vertex)

    if algorithm == 'kwik':
        kc_clustering = kwikcluster(id2vertex, graph)
        check_clustering = _check_clustering(kc_clustering, n)
        if not check_clustering:
            raise Exception('ERROR: malformed clustering')
        ret = []
        for cluster in kc_clustering:
            ret.append(_map_cluster(cluster, id2vertex))
        return np.array(ret)

    # additional algorithms
    # n = len(id2vertex)
    # m = len(edges)
    # vertex_pairs = n * (n - 1) / 2
    # vertex_triples = n * (n - 1) * (n - 2) / 6
    if algorithm == 'charikar' or algorithm == 'demaine':
        # build linear program
        # print(separator)
        # print('O(log n)-approximation algorithm - Building linear program (solver: %s)...' % (solver))
        # start = time.time()
        id2vertexpair = _vertex_pair_ids(n)
        # model = None
        # A = None
        # b = None
        # c = None
        # c_nonzero = None
        if solver == 'scipy':
            (A, b, c) = _linear_program_scipy(n, edges, graph)
            # c_nonzero = len([x for x in c if x != 0])
        # elif solver == 'pulp':
        #     model = _linear_program_pulp(n, edges, graph)
        else:
            raise Exception('Solver \'%s\' not supported' % solver)
        # runtime = _running_time_ms(start)
        # print('Linear program successfully built in %d ms' % (runtime))
        # if solver == 'scipy':
        # print('#variables: %d (must be equal to #vertex pairs, i.e., equal to %d)' % (len(c), vertex_pairs))
        # print('#inequality constraints: %d (must be equal to 3 * #vertex triples, i.e., equal to %d)' % (
        #     len(A), 3 * vertex_triples))
        # print('#non-zero entries in cost vector: %d (must be <= #edges, i.e., <= %d)' % (c_nonzero, m))

        # solving linear program
        # print(separator)
        # print('O(log n)-approximation algorithm - Solving linear program (solver: %s)...' % (solver))
        # start = time.time()
        # lp_var_assignment = None
        # obj_value = None
        # method = ''
        if solver == 'scipy':
            # method = 'SciPy'
            (lp_var_assignment, obj_value) = _solve_lp_scipy(A, b, c)
        # elif solver == 'pulp':
        #     method = 'PuLP'
        #     (lp_var_assignment, obj_value) = _solve_lp_pulp(model)
        else:
            raise Exception('Solver \'%s\' not supported' % solver)
        # runtime = _running_time_ms(start)
        lp_cost = _lp_solution_cost(lp_var_assignment, graph, n) + tot_min
        # print('Linear program successfully solved in %d ms' % (runtime))
        # print('Size of the solution array: %d (must be equal to #variables)' % (len(lp_var_assignment)))
        # print('Cost of the LP solution: %s (tot_min: %s, cost-tot_min: %s)' % (lp_cost, tot_min, lp_cost - tot_min))
        # all_negativeedgeweight_sum = _all_negativeedgeweight_sum(graph)
        # print('Cost of the LP solution (according to %s): %s (tot_min: %s, cost-tot_min: %s)' % (
        #     method, obj_value + all_negativeedgeweight_sum + tot_min, tot_min, obj_value + all_negativeedgeweight_sum))

        # rounding lp solution
        # print(separator)
        # print('O(log n)-approximation algorithm - Rounding the LP solution (rounding algorithm: %s)...' % (algorithm))
        # start = time.time()
        clustering = None
        if algorithm == 'charikar':
            clustering = round_charikar(lp_var_assignment, id2vertexpair, id2vertex, edges, graph, lp_cost - tot_min)
        elif algorithm == 'demaine':
            clustering = round_demaine(lp_var_assignment, id2vertexpair, id2vertex, edges, graph, 2 + eps)
        # runtime = _running_time_ms(start)
        check_clustering = _check_clustering(clustering, n)
        if not check_clustering:
            raise Exception('ERROR: malformed clustering')
        # print('LP-rounding successfully performed in %d ms' % (runtime))
        # cc_cost = _CC_cost(clustering, graph) + tot_min
        # print('CC cost of O(log n)-approximation algorithm\'s output clustering: %s (tot_min: %s, cost-tot_min: %s)' % (
        #     cc_cost, tot_min, cc_cost - tot_min))
        # print('O(log n)-approximation algorithm\'s output clustering:')
        # c = 1
        ret = []
        for cluster in clustering:
            # mapped_cluster = _map_cluster(cluster, id2vertex)
            # print('Cluster ' + str(c) + ': ' + str(sorted(mapped_cluster)))
            # c += 1
            ret.append(_map_cluster(cluster, id2vertex))
        return np.array(ret)

    # should never get here
    raise Exception(f'ERROR: algorithm={algorithm}')
