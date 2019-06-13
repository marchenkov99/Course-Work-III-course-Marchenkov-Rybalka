import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint
from numpy import random
import networkx as nx
import tkinter as tk
from statistics import mean
import time


def read_file(path):
    edges = []
    with open(path) as file:
        vertex = list(map(int, file.readline().split()))[0]

        weights_list = list(map(int, file.readline().split()))
        weights = {}

        for i in range(len(weights_list)):
            weights[i + 1] = weights_list[i]

        incidence = [[0 for col in range(vertex)] for row in range(vertex)]

        for line in file.readlines():
            split_line = line.rstrip().split(' ')
            split_line = [i.split('\t', 1)[0] for i in split_line]
            split_line = [int(i) for i in split_line]

            incidence[split_line[0] - 1][split_line[1] - 1] = 1
            edges.append([split_line[0], split_line[1]])

        return incidence, weights, edges


def read_data(data):
    node = int(str(data[0]))
    weights_list = list(map(int, (map(str, data[1].split()))))
    edges = list(map(int, data[2].split()))
    edgelist = []

    weights = {}
    for i in range(len(weights_list)):
        weights[i + 1] = weights_list[i]

    incidence = [[0 for col in range(node)] for row in range(node)]
    for i in range(0, len(edges), 2):
        incidence[edges[i] - 1][edges[i + 1] - 1] = 1
        edgelist.append([edges[i], edges[i + 1]])

    return incidence, weights, edgelist


def greedy(incidence, weights_dict):
    result = {}
    sorted_weights = sorted(weights_dict, key = weights_dict.get, reverse = True)

    while sorted_weights:
        current = sorted_weights.pop(0)
        result[current] = 1

        for i in range(len(incidence)):
            if incidence[current - 1][i] == 1:
                result[i + 1] = 0
                if i + 1 in sorted_weights:
                    sorted_weights.remove(i + 1)

        for i in range(len(incidence)):
            if incidence[i][current - 1] == 1:
                result[i + 1] = 0
                if i + 1 in sorted_weights:
                    sorted_weights.remove(i + 1)

    gaiety = 0
    for item in result.items():
        if item[1] == 1:
            gaiety += weights_dict[item[0]]

    return result, gaiety


def search_level(incidence, node, level, node_level):
    node_level[node] = level
    for i in range(len(incidence[node - 1])):
        if incidence[node - 1][i] == 1:
            node_level[i + 1] = node_level[node] + 1
            search_level(incidence, i + 1, node_level[node] + 1, node_level)

    return node_level


def dynamic(incedence, weights, node_level):
    INCL = {}
    EXCL = {}
    Result = {}
    height = max(node_level.values())

    for level in range(height, -1, -1):
        current_level_nodes = list({k for k, v in node_level.items() if v == level})
        for node in current_level_nodes:
            if 1 in incedence[node - 1]:
                INCL[node] = weights[node]
                EXCL[node] = 0

                for i in range(len(incedence[node - 1])):
                    if incedence[node - 1][i] == 1:
                        INCL[node] += EXCL[i + 1]
                        EXCL[node] += max(INCL[i + 1], EXCL[i + 1])
            else:
                INCL[node] = weights[node]
                EXCL[node] = 0

    for node in list(INCL):
        if INCL[node] > EXCL[node]:
            Result[node] = 1
        else:
            Result[node] = 0

    max_gaiety = None

    if INCL[1] > EXCL[1]:
        max_gaiety = INCL[1]
        Result[1] = 1
        for i in range(len(incedence[0])):
            if incedence[0][i] == 1:
                Result[i + 1] = 0

    else:
        max_gaiety = EXCL[1]
        Result[1] = 0
        for i in range(len(incedence[0])):
            if incedence[0][i] == 1:
                Result[i + 1] = 1

    return Result, max_gaiety


def draw_graph(edges, node_come):
    go_nodes = list({k for k, v in node_come.items() if v == 1})
    not_go_nodes = list({k for k, v in node_come.items() if v == 0})

    class OrderedNodeGraph(nx.Graph):
        node_dict_factory = OrderedDict
        adjlist_dict_factory = OrderedDict
    G = OrderedNodeGraph()

    G.add_nodes_from(node_come.keys())
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=go_nodes,
                           node_color='g',
                           node_size=100,
                           alpha=0.7)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=not_go_nodes,
                           node_color='r',
                           node_size=100,
                           alpha=0.7)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    labels = {}

    for i in range(1, len(node_come) + 1):
        labels[i] = r'$' + str(i) +'$'

    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    plt.axis('off')
    plt.title('Dynamic algorythm')
    plt.show()


def generate_tree(branch, height):
    G = nx.balanced_tree(branch, height)
    vertex = G.number_of_nodes()
    edgelist = G.edges()
    weightdict = {}

    incidence = [[0 for col in range(vertex)] for row in range(vertex)]
    for edge in edgelist:
        incidence[edge[0]][edge[1]] = 1

    nodelist = list(G.nodes())

    for i in range(vertex):
        weightdict[nodelist[i] + 1] = random.randint(5, 30)

    edgelist = [(edge[0] + 1, edge[1] + 1) for edge in edgelist]

    return incidence, weightdict, edgelist


def create_form():
    global s
    s = None
    global incidence_matrix
    global weights
    global edges
    incidence_matrix = []
    weights = {}
    edges = []
    form = tk.Tk()
    form.title("Maximum rating")
    form.geometry("1000x300")

    def from_file():
        global incidence_matrix
        global weights
        global edges
        incidence_matrix, weights, edges = read_file("input.txt")

    def randomly():
        global incidence_matrix
        global weights
        global edges
        incidence_matrix, weights, edges = generate_tree(3, 3)

    def manually():
        global incidence_matrix
        global weights
        global edges
        global node_message
        global  weight_message
        s = get_text()
        node_num = node_message.get()
        weight_list = weight_message.get()
        data = node_num, weight_list, s[:-1]
        incidence_matrix, weights, edges = read_data(data)

    lbl_nodes = tk.Label(form, text ='Number of nodes')
    lbl_nodes.place(relx =.1, rely =.065)
    lbl_weights = tk.Label(form, text='Weights')
    lbl_weights.place(relx=.1, rely=.16)

    global node_message
    node_message = tk.StringVar()
    node_entry = tk.Entry(textvariable=node_message, width=5)
    node_entry.place(relx=.5, rely=.1, anchor="c")

    global weight_message
    weight_message = tk.StringVar()
    weight_entry = tk.Entry(textvariable=weight_message, width=45)
    weight_entry.place(relx =.5, rely =.2, anchor = "c")

    T = tk.Text(height = 9, width = 6)
    T.pack()
    T.place(relx =.4, rely =.55, anchor="c")


    def get_text():
        global s
        s = T.get(1.0, tk.END)
        label['text'] = s[:-1]
        return s

    lbl_output = tk.Label(form)
    lbl_output.place(relx=.45, rely=.35)

    def calculate_result():
        Gaiety_greedy, Result_greedy, Gaiety_dynamic, Result_dynamic = \
            perform_operation(incidence_matrix, weights)

        lbl_res_1 = []
        lbl_res_2 = []

        for item in list(sorted(Result_greedy.items())):
            if item[1] == 1:
                lbl_res_1.append(item[0])

        for item in list(sorted(Result_dynamic.items())):
            if item[1] == 1:
                lbl_res_2.append(item[0])

        lbl_output.config(text="Greedy algorythm/>\n" + str(lbl_res_1) +
                          "\nGaiety = " + str(Gaiety_greedy) +
                          "\n\nDynamic algorythm/>\n" + str(lbl_res_2) +
                          "\nGaiety = " + str(Gaiety_dynamic))

        draw_graph(edges, Result_dynamic)

    var = tk.StringVar()

    R1 = tk.Radiobutton(text="Enter manually", variable = var, value='Enter manually',
                        command=manually)
    R1.place(relx=.05, rely=.3)
    R2 = tk.Radiobutton(text="Read from file", variable = var, value='Read from file',
                        command=from_file)
    R2.place(relx=.05, rely=.4)
    R3 = tk.Radiobutton(text="Randomly generate", variable = var, value='Randomly generate',
                        command=randomly)
    R3.place(relx=.05, rely=.5)

    submit = tk.Button(text="Solve", command=calculate_result, padx="20", pady="8")
    submit.place(relx=.5, rely=.9, anchor="c", height=30, width=100)

    label = tk.Label()
    form.mainloop()


def perform_operation(incidence_matrix, weights):
    level_dict = {}
    Result_greedy, Gaiety_greedy = greedy(incidence_matrix, weights)
    node_level = search_level(incidence_matrix, 1, 0, level_dict)
    Result_dynamic, Gaiety_dynamic = dynamic(incidence_matrix, weights, node_level)

    return Gaiety_greedy, Result_greedy, Gaiety_dynamic, Result_dynamic


if __name__ == '__main__':
    create_form()
    """
    x = []
    greedy_time = []
    dynamic_time = []

    counter = 1
    for i in range(2, 5):
        #if i==3:break
        for j in range(2, 5):
            if j==3: break
            for k in range(25):
                level_dict = {}
                incidence_matrix, weights, edge_list = generate_tree(i, j)
                start_greedy = time.time()
                for n in range(400):
                    Result_greedy, Gaiety_greedy = greedy(incidence_matrix, weights)
                finish_greed = time.time() - start_greedy

                start_dynamic = time.time()
                node_level = search_level(incidence_matrix, 1, 0, level_dict)
                for n in range(1000):
                    Result_dynamic, Gaiety_dynamic = dynamic(incidence_matrix, weights, node_level)
                finish_dynamic = time.time() - start_dynamic

                x.append(counter)
                counter += 1
                greedy_time.append(finish_greed)
                dynamic_time.append(finish_dynamic)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, greedy_time, alpha=0.8, c="blue", label="Greedy algorithm")
    ax.plot(x, dynamic_time, alpha=0.8, c="green", label="Dynamic algorithm")
    plt.xlabel('Nodes')
    plt.ylabel('Time')
    #plt.axhline(mean(greedy_time), color='blue')
    #plt.axhline(mean(finish_dynamic), color='green')
    plt.legend(loc=2)
    plt.show()
    """