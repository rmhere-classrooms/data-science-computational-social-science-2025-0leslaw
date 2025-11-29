library(igraph)

# zad 2
set.seed(123)
g_ba <- barabasi.game(
  n = 1000,
  power = 1,
  m = 1,
  directed = FALSE
)

# zad 3

layout_fr <- layout_with_fr(g_ba)

plot(
  g_ba,
  layout = layout_fr,
  vertex.size = 3,
  vertex.label = NA,
  edge.width = 0.5,
  main = "Zad 3"
)

# zad 4
cat("zad4\n\n")

bet <- betweenness(g_ba)
most_central_node <- which.max(bet)

cat("najbardziej centralny węzeł:\n")
cat(most_central_node, "\n")
cat("Wartość betweenness:", bet[most_central_node], "\n")

# zad 5
graph_diameter <- diameter(g_ba)
cat("zad5\n\n")
cat("Średnica:", graph_diameter, "\n")
    
# zad 6
# Graf Barabási–Albert różni się od grafu Erdős–Rényi tym, 
# że powstaje w procesie wzrostu oraz preferencyjnego
# dołączania nowych węzłów do już istniejących o dużym stopniu.
# W rezultacie tworzą się węzły-huby o bardzo dużej liczbie
# połączeń, w grafie Erdős–Rényi połączenia powstają z jednakowym 
# prawdopodobieństwem z tego względu nie powstają węzły-huby
