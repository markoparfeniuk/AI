import random
import matplotlib.pyplot as plt

# N = random.randint(25, 35)
# cities = range(N)
# distances = [[random.randint(10, 100) for _ in cities] for _ in cities]


# Save the map to a file
# with open('map.txt', 'w') as f:
#     f.write(f'{N}\n')
#     for row in distances:
#         f.write(' '.join(map(str, row)) + '\n')

# Load the map from a file
with open('map.txt', 'r') as f:
    N = int(f.readline())
    cities=range(N)
    distances = [list(map(int, line.split())) for line in f]

# Set the parameter values to test
M_values = [10, 50, 100]
rho_values = [0.3, 0.6, 0.9]
alpha_values = [0.5, 1.5, 2.0]
beta_values = [2.5, 1.5, 1.0]

# Ant algorithm parameters
M = 50
alpha = 1.0
beta = 5.0
rho = 0.5
Q = 100

# Initialize pheromone levels
pheromones = [[1 / (N * N) for _ in cities] for _ in cities]

def length(tour):
    # Calculate the length of a tour
    return sum(distances[tour[i - 1]][tour[i]] for i in cities)

def transition_probability(alpha, beta, visited):
    # Calculate the transition probabilities for ant k
    unvisited = set(cities) - set(visited)
    denominator = sum(pheromones[visited[-1]][j] ** alpha * (1 / distances[visited[-1]][j]) ** beta for j in unvisited)
    probabilities = [pheromones[visited[-1]][j] ** alpha * (1 / distances[visited[-1]][j]) ** beta / denominator if j in unvisited else 0 for j in cities]
    return probabilities

def construct_solution(M, N, alpha, beta):
    # Construct a solution using the ant algorithm
    tours = [[] for _ in range(M)]
    for k in range(M):
        tours[k].append(random.randint(0, N - 1))
        while len(tours[k]) < N:
            probabilities = transition_probability(alpha, beta, tours[k])
            next_city = random.choices(cities, weights=probabilities)[0]
            tours[k].append(next_city)
        tours[k].append(tours[k][0])
    return tours

def update_pheromones(tours, M, N, rho):
    # Update pheromone levels
    L = [length(tour) for tour in tours]
    delta_pheromones = [[0 for _ in cities] for _ in cities]
    for k in range(M):
        for i in range(N):
            delta_pheromones[tours[k][i]][tours[k][i + 1]] += Q / L[k]
    for i in cities:
        for j in cities:
            pheromones[i][j] = (1 - rho) * pheromones[i][j] + delta_pheromones[i][j]

def plot_tour(tour):
    # Plot a tour
    x = [distances[tour[i]][0] for i in range(N + 1)]
    y = [distances[tour[i]][1] for i in range(N + 1)]
    plt.plot(x, y, 'o-')
    for i in range(N + 1):
        plt.annotate(str(i + 1), (x[i], y[i]))
    plt.show()

# Conduct a sequence of simulations on the same map
for M in M_values:
    pheromones = [[1 / (N * N) for _ in cities] for _ in cities]
    for step in range(50):
        tours = construct_solution(M, N, alpha, beta)
        update_pheromones(tours, M, N, rho)
    best_tour = min(tours,key=length)
    best_length = length(best_tour)
    print(f'M: {M}, rho: {rho}, alpha: {alpha}, beta: {beta}')
    print(f'Best tour: {best_tour}')
    print(f'Best length: {best_length}')

for rho in rho_values:
    pheromones = [[1 / (N * N) for _ in cities] for _ in cities]
    for step in range(50):
        tours = construct_solution(M, N, alpha, beta)
        update_pheromones(tours, M, N, rho)
    best_tour = min(tours,key=length)
    best_length = length(best_tour)
    print(f'M: {M}, rho: {rho}, alpha: {alpha}, beta: {beta}')
    print(f'Best tour: {best_tour}')
    print(f'Best length: {best_length}')
    plot_tour(best_tour)

for alpha, beta in zip(alpha_values, beta_values):
    # Initialize pheromone levels
    pheromones = [[1 / (N * N) for _ in cities] for _ in cities]
    for step in range(50):
        tours = construct_solution(M, N, alpha, beta)
        update_pheromones(tours, M, N, rho)
    best_tour = min(tours,key=length)
    best_length = length(best_tour)
    print(f'M: {M}, rho: {rho}, alpha: {alpha}, beta: {beta}')
    print(f'Best tour: {best_tour}')
    print(f'Best length: {best_length}')
    plot_tour(best_tour)
