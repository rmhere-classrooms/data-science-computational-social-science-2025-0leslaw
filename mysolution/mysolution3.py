from collections import defaultdict
import enum
from itertools import combinations
import pandas as pd
import igraph as ig
import random
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import io
import base64

# Aby odpalic lokalnie nalezy w dir / odpalic:
# python -m venv .venv
# .venv\Scripts\activate
# pip install -r requirements.txt

# Odpalanie aplikacji:
# cd mysolution
# python -m shiny run mysolution3.py

# lub jeśli chcemy policzyć sobie w pythonie to po prostu bez shiny: 
# python mysolution3.py 


app_ui = ui.page_fluid(
    ui.h2("Interactive Matplotlib Plot with Shiny Python"),
    ui.input_slider("probability_boost", "Probability Boost", 50, 200, 100),
    ui.input_slider("max_steps", "Max Steps", 1, 50, 10),
    ui.output_plot("plot")
)

def server(input, output, session): 
    
    @reactive.Calc
    def plot_figure():
        graph = define_graph()
        fig = run_experiments(graph, num_runs=20, max_steps=input.max_steps(), proportion_init_active=0.05, probability_boost=input.probability_boost()/100)
        fig.tight_layout()
        return fig
    
    @output
    @render.plot
    def plot():
        fig = plot_figure()
        return fig

app = App(app_ui, server)



  
# helpers
def define_graph():
    df = pd.read_csv("./out.radoslaw_email_email", header=None, sep=r"\s+")
    df = df.iloc[2:, [0,1]]  
    df = df.rename(columns={0: "origin", 1: "destination"})
    df["origin"] = df["origin"].astype(int)
    df["destination"] = df["destination"].astype(int)



    all_ids = pd.concat([df["origin"], df["destination"]]).unique()

    graph = ig.Graph(directed=True)
    graph.add_vertices(len(all_ids))

    graph.vs["name"] = all_ids.tolist()

    edges = [(row["origin"], row["destination"]) for _, row in df.iterrows()]
    name_to_vid = {v["name"]: v.index for v in graph.vs}
    edges_internal = [(name_to_vid[src], name_to_vid[dst]) for src, dst in edges]

    graph.add_edges(edges_internal)

    # zad 5
    graph = graph.simplify(multiple=True, loops=True, combine_edges=None)

    # zad 6
    name_to_vid = {v["name"]: v.index for v in graph.vs}

    pair_counts = df.groupby(["origin", "destination"]).size().reset_index(name="cntij")

    origin_counts = df.groupby("origin").size().to_dict()

    pair_counts["weight"] = pair_counts.apply(
        lambda row: row["cntij"] / origin_counts[row["origin"]], axis=1
    )

    graph.es["weight"] = 0.0

    name_to_vid = {v["name"]: v.index for v in graph.vs}

    for _, row in pair_counts.iterrows():
        u_name = row["origin"]
        v_name = row["destination"]
        weight = row["weight"]
        
        if u_name in name_to_vid and v_name in name_to_vid:
            u_id = name_to_vid[u_name]
            v_id = name_to_vid[v_name]
            
            # Get edge ID
            eid = graph.get_eid(u_id, v_id, directed=True, error=False)
            if eid != -1:
                graph.es[eid]["weight"] = weight


    for vertex in graph.vs:
        vertex["activated"] = False 
        vertex["tried_act"] = False
    return graph
        
        
class SimulationStart(enum.Enum):
    HIGHEST_OUTDEGREE = 1
    HIGHEST_BETWEENNESS = 2
    HIGHEST_CLOSENESS = 3
    RANDOM = 4
    SAME_FROM_DIFFERENT_CLUSTERS_IF_POSSIBLE = 5
        
        

def run_simulation(graph, pick_start_by: SimulationStart, proportion_init_active: float=0.05, max_steps: int=10, probability_boost: float=1.0):
    g = graph.copy()
    num_vertices = g.vcount()
    num_initial_active = max(1, int(proportion_init_active * num_vertices))
    
    if pick_start_by == SimulationStart.HIGHEST_OUTDEGREE:
        out_degrees = g.outdegree()
        initial_active_vertices = sorted(range(num_vertices), key=lambda i: out_degrees[i], reverse=True)[:num_initial_active]
    elif pick_start_by == SimulationStart.HIGHEST_BETWEENNESS:
        betweenness = g.betweenness()
        initial_active_vertices = sorted(range(num_vertices), key=lambda i: betweenness[i], reverse=True)[:num_initial_active]
    elif pick_start_by == SimulationStart.HIGHEST_CLOSENESS:
        closeness = g.closeness()
        initial_active_vertices = sorted(range(num_vertices), key=lambda i: closeness[i], reverse=True)[:num_initial_active]
    elif pick_start_by == SimulationStart.RANDOM:
        initial_active_vertices = random.sample(range(num_vertices), num_initial_active)
    elif pick_start_by == SimulationStart.SAME_FROM_DIFFERENT_CLUSTERS_IF_POSSIBLE:
        # Ta metoda wybiera wierzchołki z różnych klastrów po równo,
        # jeśli liczba wybranych wierzchołków jest mniejsza od liczby wymaganych 
        # wierzchołków, wybiera losowo spośród pozostałych.
        
        clusters = g.clusters()
        cluster_indices = list(range(len(clusters)))
        take_from_clusters = min(num_initial_active // len(clusters), min([len(c) for c in clusters]))
        initial_active_vertices = []
        while len(initial_active_vertices) < num_initial_active and cluster_indices:
            cluster_index = cluster_indices.pop(0)
            cluster_vertices = clusters[cluster_index]
            # sort cluster vertices by betweenness to pick the most central ones
            cluster_vertices = sorted(cluster_vertices, key=lambda i: g.betweenness()[i], reverse=True)
            if cluster_vertices:
                initial_active_vertices.extend(cluster_vertices[:take_from_clusters])
        if len(initial_active_vertices) < num_initial_active:
            remaining = num_initial_active - len(initial_active_vertices)
            all_vertices = set(range(num_vertices))
            already_chosen = set(initial_active_vertices)
            available_vertices = list(all_vertices - already_chosen)
            initial_active_vertices.extend(random.sample(available_vertices, remaining))
    
    for v in initial_active_vertices:
        g.vs[v]["activated"] = True
    
    activated = initial_active_vertices 
    activated_count_history = [len(activated)]
    while activated and len(activated_count_history) - 1 < max_steps:
        new_activated = []
        for v in activated:
            neighbors = g.neighbors(v, mode="out")
            for neighbor in neighbors:
                if not g.vs[neighbor]["activated"]:
                    edge_id = g.get_eid(v, neighbor)
                    activation_prob = g.es[edge_id]["weight"]
                    if random.random() < activation_prob * probability_boost:
                        g.vs[neighbor]["activated"] = True
                        new_activated.append(neighbor)
        activated_count_history.append(sum(v["activated"] for v in g.vs))
        activated = new_activated
    return activated_count_history


def run_experiments(graph, num_runs: int=100, max_steps: int=10, proportion_init_active: float=0.05, probability_boost: float=1.0):
    all_histories = []
    simulation_start_to_results = {}
    for enum_start in SimulationStart:
        aggregated_history = [0] * (max_steps + 1)
        for _ in range(num_runs):
            history = run_simulation(graph, pick_start_by=enum_start, max_steps=max_steps, proportion_init_active=proportion_init_active, probability_boost=probability_boost)
            for step, count in enumerate(history):
                aggregated_history[step] += count
        averaged_history = [count / num_runs for count in aggregated_history]
        simulation_start_to_results[enum_start] = averaged_history
    
    fig, ax = plt.subplots(figsize=(8, 5)) 

    for enum_start, history in simulation_start_to_results.items():
        ax.plot(history, label=enum_start.name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Average Activated Count")
    ax.set_title("Simulation Results by Start Strategy")
    ax.legend()
    fig.tight_layout()
    
    return fig 

if __name__ == "__main__":
    graph = define_graph()
    run_experiments(graph, num_runs=20, max_steps=15, proportion_init_active=0.05, probability_boost=1.0)
    