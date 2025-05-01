import random
import time
import os
import shutil
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
size = 15
red_requirement = 5
black_requirement = 4
vert_amount = int((size * (size - 1)) / 2)

# === CONFIG ===
POP_SIZE = 20
GENERATIONS = 1000
CXPB = 0.2  # Crossover probability
MUTPB = 0.5  # Mutation probability


#########################################################################
######################## TOOLS ##########################################
#########################################################################

def move_previous_results():
    # Get the current working directory
    current_dir = os.getcwd()

    # Find the .txt file (expecting only one)
    txt_files = [f for f in os.listdir(current_dir) if f.endswith(".txt")]

    if len(txt_files) != 1:
        raise Exception("Expected exactly one .txt file in the directory.")

    # Get the base name (without .txt extension)
    txt_file = txt_files[0]
    folder_name = os.path.splitext(txt_file)[0]
    destination_path = os.path.join(current_dir, folder_name)

    # Create the folder if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # Move the .txt file
    shutil.move(os.path.join(current_dir, txt_file), os.path.join(destination_path, txt_file))
    print(f"Moved {txt_file} to {folder_name}")

    # Move all .png files to that folder
    for filename in os.listdir(current_dir):
        if filename.endswith(".png"):
            shutil.move(os.path.join(current_dir, filename), os.path.join(destination_path, filename))
            print(f"Moved {filename} to {folder_name}")
def graph_gen(_vert_amount):
    return [random.randint(0, 1) for _ in range(_vert_amount)]


#------------------------------------------------------------------------
def binatodeci(binary):
    number = 0
    for b in binary:
        number = (2 * number) + b
    return number


#------------------------------------------------------------------------
def decitobina(decimal):
    number = bin(decimal)[2:]
    number = [int(d) for d in number]
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
def translate_vertices(randomized_vertices):
    # Here generated 0 and 1s are translated into two graphs, for now as list of edge connections.
    black_edges = []
    red_edges = []

    current_node = 0
    connected_nodes = size - 1
    sub_iterator = 0
    for i in range(0, len(randomized_vertices)):

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
# Draws a graph with highlighted small_graph - used to show the biggest clique found in big_graph.
# If there was no smaller graph (in broader terms, argument for this number was met)
def draw_overlay_graph(graphId, big_graph, small_graph, color):
    pos = nx.circular_layout(big_graph)
    if small_graph is not None:
        if color == 1:
            nx.draw(big_graph, pos, with_labels=True, edge_color="red", node_size=500)
            nx.draw(small_graph, pos, with_labels=True, edge_color="green", node_size=500)
        else:
            nx.draw(big_graph, pos, with_labels=True, edge_color="black", node_size=500)
            nx.draw(small_graph, pos, with_labels=True, edge_color="navy", node_size=500)
    else:
        if color == 1:
            nx.draw(big_graph, pos, with_labels=True, edge_color="red", node_size=500)
        else:
            nx.draw(big_graph, pos, with_labels=True, edge_color="black", node_size=500)



    fig = plt.gcf()
    fig.patch.set_facecolor('lightgray')
    # plt.figure(figsize=(8, 6), dpi=80)
    if color == 1:
        plt.savefig(f'{graphId}_red_graph.png')
    else:
        plt.savefig(f'{graphId}_black_graph.png')
    plt.show()


#------------------------------------------------------------------------
# Draws both biggest cliques, red and black, on top of full graph of given size.
def draw_final_graph(graphId, big_graph, red_graph, black_graph):
    pos = nx.circular_layout(big_graph)

    #Drawing the main, full graph - the cycles will be shown on top of it
    nx.draw(big_graph, pos, with_labels=True, edge_color="green", node_size=500)

    if red_graph is not None:
        nx.draw(red_graph, pos, with_labels=True, edge_color="black", node_size=500)

    if black_graph is not None:
        nx.draw(black_graph, pos, with_labels=True, edge_color="red", node_size=500)

    # fig = plt.gcf()
    # fig.patch.set_facecolor('gray')
    # plt.figure(figsize=(8, 6), dpi=80)
    plt.savefig(f'{graphId}_full_graph.png')
    plt.show()


#########################################################################
######################## MACHINE LEARNING ###############################
#########################################################################
def graph_validator(tested_edges, required_size):
    score = [0.0, None]

    G = nx.Graph()
    G.add_nodes_from((lambda i: list(range(0, i)))(size))
    G.add_edges_from(tested_edges)

    #TODO: Partition this so amount of subgraphs is set, currently memory error, there're too many
    subgraphs = extract_subgraphs(G, required_size, size)
    # subgraphs = extract_subgraphs(G, required_size, size)


    # biggest_subgraph = nx.Graph()
    max_found_size = 0

    for subgraph in subgraphs:
        if is_complete_graph(subgraph):
            if len(subgraph) > max_found_size:
                print(len(subgraph))
                max_found_size = len(subgraph)
                score[1] = subgraph

    if max_found_size < required_size:
        score[0] = 1.0
    else:
        score[0] = required_size/(max_found_size + 1)

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
    black_verts, red_verts = translate_vertices(test_subject)

    # TODO : Simple foundSize/requiredSize promotes smaller requirements
    black_result = graph_validator_noGraph(black_verts, black_requirement)
    red_result = graph_validator_noGraph(red_verts, red_requirement)

    return (black_result[0] + red_result[0],)


#------------------------------------------------------------------------
def run_ga():
    move_previous_results()
    return evolve_and_track(POP_SIZE, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS)


#------------------------------------------------------------------------
def evolve_and_track(pop_size, toolbox, cxpb, mutpb, ngen):
    population = toolbox.population(n=pop_size)
    current_time = datetime.now()
    alg_start_time = time.time()
    best_fitness = 0.0

    current_time_str = current_time.strftime("%Y-%H-%M")
    with open(f'{current_time_str}_results.txt', 'a') as file:
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        best = tools.selBest(population, k=1)[0]
        string_output = f"Generation 1: Best = {binatodeci(best)}, Fitness = {best.fitness.values[0]}\n"
        print(string_output)
        file.write(string_output)


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

            # Get best of current generation
            best = tools.selBest(population, k=1)[0]

            elapsed_time = time.time() - start_time

            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int((elapsed_time % 1) * 1000)

            formatted_time = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
            string_output = f"Generation {gen + 1}: Best = {binatodeci(best): .5f}, Fitness = {best.fitness.values[0]}\nGeneration time: {formatted_time}"

            if best.fitness.values[0] > best_fitness:
                draw_graphs_from_index(best)
                string_output = f"\nFitness change at generation {gen + 1} from {best_fitness: .5f} to {best.fitness.values[0]: .5f}\n{string_output}"
                print(string_output)
                file.write(string_output)
                best_fitness = best.fitness.values[0]

            else:
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
def draw_graphs_from_index(raw_vertices):
    black_edges, red_edges = translate_vertices(raw_vertices)

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

        if black_result[1] is None and red_result[1] is None:
            draw_final_graph(graphId, nx.complete_graph(size), black_graph, red_graph)
        else:
            draw_final_graph(graphId, nx.complete_graph(size), black_result[1], red_result[1])
        print("black result:", black_result[0])
        print("red result:", red_result[0])


#------------------------------------------------------------------------
if __name__ == '__main__':

    Printing_Graphs = False

    # 0 = black, 1 = red
    if Printing_Graphs:
        test = 6898044836609178785
        raw_vertices = decitobina(test)
        draw_graphs_from_index(raw_vertices)

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

        draw_graphs_from_index(best_graph)
