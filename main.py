import random
import time

import networkx.algorithms.clique
import rustworkx as rx
import rustworkx.generators as generators
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyparsing import nums
from rustworkx import networkx_converter
from rustworkx.visualization.matplotlib import mpl_draw
import networkx as nx
from itertools import combinations, chain
import igraph as ig
import pygad
from datetime import datetime

from deap import base, creator, tools, algorithms

plt.rcParams["figure.figsize"] = (4, 3)
# 43-46
size = 7
red_requirement = 3
black_requirement = 4
vert_amount = int((size * (size - 1)) / 2)

# === CONFIG ===
POP_SIZE = 100
GENERATIONS = 100
CXPB = 0.5  # Crossover probability
MUTPB = 0.35  # Mutation probability


#########################################################################
######################## TOOLS ##########################################
#########################################################################

def binatodeci(binary):
    number = 0
    for b in binary:
        number = (2 * number) + b
    return number


#------------------------------------------------------------------------
def extract_subgraphs(G, min_size, max_size):
    node_combinations = list(chain.from_iterable(combinations(G.nodes, r) for r in range(min_size, max_size)))
    # node_combinations = list(chain.from_iterable(combinations(G.nodes, r) for r in range(2, min_size + 1)))

    # Generate subgraphs
    subgraphs = [G.subgraph(n).copy() for n in node_combinations]

    return subgraphs


#------------------------------------------------------------------------
def is_complete_graph(G):
    N = len(G) - 1
    return not any(n in nbrdict or len(nbrdict) != N for n, nbrdict in G.adj.items())


#------------------------------------------------------------------------
def translate_vertices(randomized_vertices, vert_amount):
    # Here generated 0 and 1s are translated into two graphs, for now as list of edge connections.
    black_edges = []
    red_edges = []

    current_node = 0
    connected_nodes = size - 1
    sub_iterator = 0
    for i in range(0, vert_amount):

        sub_iterator += 1

        # shouldn't ever trigger
        if connected_nodes == 0:
            print("Error in edge assignment - loop didn't finish before connected_nodes hit 0")
            break

        if int(randomized_vertices[i]) == 1:
            black_edges.append((current_node, sub_iterator + current_node))
        else:
            red_edges.append((current_node, sub_iterator + current_node))

        if sub_iterator == connected_nodes:
            connected_nodes -= 1
            sub_iterator = 0
            current_node += 1

    return black_edges, red_edges


#------------------------------------------------------------------------
def draw_overlay_graph(graphId, big_graph, small_graph, color):
    pos = nx.circular_layout(big_graph)
    if color == 1:
        nx.draw(big_graph, pos, with_labels=True, edge_color="green", node_size=500)
        nx.draw(small_graph, pos, with_labels=True, edge_color="red", node_size=500)
    else:
        nx.draw(big_graph, pos, with_labels=True, edge_color="green", node_size=500)
        nx.draw(small_graph, pos, with_labels=True, edge_color="black", node_size=500)

    # fig = plt.gcf()
    # fig.patch.set_facecolor('gray')
    # plt.figure(figsize=(8, 6), dpi=80)
    if color == 1:
        plt.savefig(f'{graphId}_red_graph.png')
    else:
        plt.savefig(f'{graphId}_black_graph.png')
    plt.show()


#------------------------------------------------------------------------
def draw_final_graph(graphId, big_graph, red_graph, black_graph):
    pos = nx.circular_layout(big_graph)

    nx.draw(big_graph, pos, with_labels=True, edge_color="green", node_size=500)

    nx.draw(red_graph, pos, with_labels=True, edge_color="black", node_size=500)

    nx.draw(black_graph, pos, with_labels=True, edge_color="red", node_size=500)

    # fig = plt.gcf()
    # fig.patch.set_facecolor('gray')
    # plt.figure(figsize=(8, 6), dpi=80)
    plt.savefig(f'{graphId}_full_graph.png')
    plt.show()


#########################################################################
######################## MACHINE LEARNING ###############################
#########################################################################

def graph_gen(vert_amount):
    return [random.randint(0, 1) for _ in range(vert_amount)]


#------------------------------------------------------------------------
def graph_validator(tested_edges, required_size):
    score = [0.0, None]

    G = nx.Graph()
    G.add_nodes_from((lambda i: list(range(0, i)))(size))
    G.add_edges_from(tested_edges)

    subgraphs = extract_subgraphs(G, 2, size)
    # subgraphs = extract_subgraphs(G, required_size, size)


    biggest_subgraph = nx.Graph()
    max_found_size = 0

    for subgraph in subgraphs:
        if is_complete_graph(subgraph):
            if len(subgraph) > max_found_size:
                print(len(subgraph))
                max_found_size = len(subgraph)
                biggest_subgraph = subgraph

    if max_found_size < required_size:
        score[0] = 1.0
    else:
        score[0] = required_size/(max_found_size + 1)

    score[1] = biggest_subgraph
    return score


#------------------------------------------------------------------------
def graph_validator_noGraph(tested_edges, required_size):
    score = [0.0,]
    G = nx.Graph()
    G.add_nodes_from((lambda i: list(range(0, i)))(required_size))

    G.add_edges_from(tested_edges)

    subgraphs = extract_subgraphs(G, required_size, G.size())

    max_found_size = 0.0

    for subgraph in subgraphs:
        if is_complete_graph(subgraph):
            if len(subgraph) > max_found_size:
                max_found_size = len(subgraph)

    if max_found_size < required_size:
        score[0] = 1.0
    else:
        score[0] = pow(required_size/(max_found_size + 1), 2)

    return score


#------------------------------------------------------------------------
def fitness_fn(test_subject):
    black_verts, red_verts = translate_vertices(test_subject, vert_amount)

    # TODO : Simple foundSize/requiredSize promotes smaller requirements
    black_result = graph_validator_noGraph(black_verts, black_requirement)
    red_result = graph_validator_noGraph(red_verts, red_requirement)

    return (black_result[0] + red_result[0],)


#------------------------------------------------------------------------
def run_ga():
    return evolve_and_track(POP_SIZE, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS)


#------------------------------------------------------------------------
def evolve_and_track(pop_size, toolbox, cxpb, mutpb, ngen):
    population = toolbox.population(n=pop_size)
    current_time = datetime.now()
    alg_start_time = time.time()

    current_time_str = current_time.strftime("%Y-%H-%M")
    with open(f'{current_time_str}_results.txt', 'a') as file:
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit


        for gen in range(ngen):
            start_time = time.time()
            # Variation: crossover and mutation
            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

            # Evaluate invalid offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Selection
            population = toolbox.select(offspring, k=len(population))

            # Print best of current generation
            best = tools.selBest(population, k=1)[0]

            elapsed_time = time.time() - start_time

            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int((elapsed_time % 1) * 1000)

            formatted_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
            string_output = f"Generation {gen + 1}: Best = {binatodeci(best)}, Fitness = {best.fitness.values[0]}\nGeneration time: {formatted_time}"

            print(string_output)
            file.write(string_output)


            if best.fitness.values[0] == 2.0:
                elapsed_time = time.time() - alg_start_time

                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                milliseconds = int((elapsed_time % 1) * 1000)

                formatted_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"

                file.write(f"\nGeneration {gen + 1} achieved success.\nBest = {binatodeci(best)}\nRaw code: {best}\nElapsed time total: {formatted_time}")
                break

    top_ind = tools.selBest(population, k=1)[0]
    return top_ind, toolbox.evaluate(top_ind)

#------------------------------------------------------------------------
def draw_graphs_from_index(raw_vertices, vert_amount):
    black_edges, red_edges = translate_vertices(raw_vertices, vert_amount)

    if black_edges is not None and red_edges is not None:
        graphId = binatodeci(raw_vertices)
        black_result = graph_validator(black_edges, black_requirement)
        red_result = graph_validator(red_edges, red_requirement)

        black_graph = nx.generators.empty_graph(size)
        black_graph.add_edges_from(black_edges)
        red_graph = nx.generators.empty_graph(size)
        red_graph.add_edges_from(red_edges)

        draw_overlay_graph(graphId, black_graph, black_result[1], 0)
        draw_overlay_graph(graphId, red_graph, red_result[1], 1)
        draw_final_graph(graphId, nx.complete_graph(size), black_result[1], red_result[1])
        print("black result:", black_result[0])
        print("red result:", red_result[0])


#------------------------------------------------------------------------
if __name__ == '__main__':

    TEMPORARY_PRINTING_BOOLEAN = False
    # 0 = black, 1 = red
    if TEMPORARY_PRINTING_BOOLEAN:
        raw_vertices = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
                        0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]
        draw_graphs_from_index(raw_vertices, vert_amount)

    else:
        # === SETUP DEAP ===
        random.seed()

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, vert_amount)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", fitness_fn)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / vert_amount)
        toolbox.register("select", tools.selTournament, tournsize=3)

        best_graph, best_graph_score = run_ga()

        print("Best graph score: ", best_graph_score, "\n", "best graph index: ", best_graph)

        draw_graphs_from_index(best_graph, vert_amount)
