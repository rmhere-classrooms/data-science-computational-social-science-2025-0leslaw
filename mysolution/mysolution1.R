library(igraph)


# zad 2
g <- erdos.renyi.game(n = 100, p.or.m = 0.05, type = "gnp", directed = FALSE)

# zad 3
cat("zad 3\n")
print(summary(g))
cat("Is weighted ", is.weighted(g), "\n\n")

# zad 4
cat("zad 4\n")
cat("verts:\n")
print(V(g))

cat("edges:\n")
print(E(g))

# zad 5
E(g)$weight <- runif(ecount(g), min = 0.01, max = 1)

#zad 6
cat("zad6\n")
print(summary(g))
cat("Is weighted ", is.weighted(g), "\n\n")

# zad 7
deg <- degree(g)

cat("zad7\n")
print(deg)

hist(
  deg,
  main = "Histogram stopni węzłów",
  xlab = "Stopień węzła",
  ylab = "Liczba węzłów",
  col = "lightblue",
  border = "black"
)

# zad 8
components_g <- components(g)
cat("zad 8\n", components_g$no, "\n")

# zad 9
pr <- page_rank(g)$vector

plot(
  g,
  vertex.size = 5 + 50 * pr,    # skalowanie rozmiaru
  vertex.label = NA,
  edge.width = E(g)$weight,
  main = "zad 9"
)
