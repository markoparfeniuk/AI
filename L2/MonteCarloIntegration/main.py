import random
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# Define the range for the Monte Carlo integration
lower_bound = 1
upper_bound = 2

# Define the types of functions
class FunctionType:
    TEST = "test",
    MAIN = "main"

# Test function for Monte Carlo integration
def test_function(x):
    return x ** 2

# Main function for Monte Carlo integration
def main_function(x):
    return np.exp(x ** 2)

# Generate a random point within the integration range and below the maximum function value
def generate_point(function):
    x = random.uniform(lower_bound, upper_bound)
    y = random.uniform(0, calculate_value(upper_bound, function))
    return (x, y)

# Calculate the value of the specified function at x
def calculate_value(x, function):
    if function == FunctionType.TEST:
        return test_function(x)
    elif function == FunctionType.MAIN:
        return main_function(x)

# Perform Monte Carlo integration with n points for the specified function
def monte_carlo_integration(n, function):
    count = 0
    points_in = []
    points_out = []
    for i in range(n):
        x, y = generate_point(function)
        if y <= calculate_value(x, function):
            count += 1
            points_in.append((x, y))
        else:
            points_out.append((x, y))
    integral = (upper_bound - lower_bound) * calculate_value(upper_bound, function) * (count / n)
    return integral, points_in, points_out

# Visualize the Monte Carlo integration with a scatter plot of points and the function graph
def visualize(points_in, points_out, function):
    plt.title(f"Monte Carlo integration for the {function} function")
    plt.scatter([p[0] for p in points_in], [p[1] for p in points_in], color='purple')
    plt.scatter([p[0] for p in points_out], [p[1] for p in points_out], color='orange')
    x = np.linspace(lower_bound, upper_bound)
    if function == FunctionType.TEST:
        plt.plot(x, test_function(x), color='blue')
    elif function == FunctionType.MAIN:
        plt.plot(x, main_function(x), color='blue')
    plt.show()

# Calculate the absolute and relative error between the estimated and exact values
def calculate_error(estimated_value, exact_value):
    absolute_error = abs(exact_value - estimated_value)
    relative_error = absolute_error / exact_value
    return (absolute_error, relative_error)


n = 2000  # Number of points to use for Monte Carlo integration

# Calculate the exact value of the test function integral
exact_test_value = (upper_bound ** 3 - lower_bound ** 3) / 3
# Perform Monte Carlo integration for the test function
estimated_test_value, points_test_in, points_test_out = monte_carlo_integration(n, FunctionType.TEST)
# Calculate the error between the estimated and exact values
absolute_error_test, relative_error_test = calculate_error(estimated_test_value, exact_test_value)

# Print the results for the test function
print("Test function.")
print(f"Exact integral value: {exact_test_value}")
print(f"Monte Carlo integration value: {estimated_test_value}")
print(f"Absolute error: {absolute_error_test}")
print(f"Relative error: {relative_error_test}")

# Visualize the Monte Carlo integration for the test function
visualize(points_test_in, points_test_out, FunctionType.TEST)

# Calculate the exact value of the main function integral
exact_main_value = np.sqrt(np.pi) / 2 * (scipy.special.erfi(upper_bound) - scipy.special.erfi(lower_bound))
# Perform Monte Carlo integration for the main function
estimated_main_value, points_main_in, points_main_out = monte_carlo_integration(n, FunctionType.MAIN)
# Calculate the error between the estimated and exact values
absolute_error_main, relative_error_main = calculate_error(estimated_main_value, exact_main_value)

# Print the results for the main function
print("\nMain function.")
print(f"Exact integral value: {exact_main_value}")
print(f"Monte Carlo integration value: {estimated_main_value}")
print(f"Absolute error: {absolute_error_main}")
print(f"Relative error: {relative_error_main}")

# Visualize the Monte Carlo integration for the main function
visualize(points_main_in, points_main_out, FunctionType.MAIN)
