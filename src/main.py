import numpy as np
from data_preprocessing import load_and_preprocess_data
from genetic_algorithm import GeneticAlgorithm
from evaluation import plot_fitness_progress, plot_feature_selection

def main():
    # Parameters
    population_size = 50
    generations = 30
    mutation_rate = 0.01
    crossover_rate = 0.8
    elitism_count = 2
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data('../data/mushrooms.csv')
    chromosome_length = len(feature_names)
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size, chromosome_length, mutation_rate, crossover_rate, elitism_count)
    population = ga.initialize_population()
    
    # Track progress
    best_fitness_history = []
    avg_fitness_history = []
    
    # Run evolution
    for generation in range(generations):
        # Evaluate population
        fitness_scores = ga.evaluate_population(population, X_train, y_train, X_test, y_test)
        
        # Track statistics
        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
        
        # Evolve population
        population = ga.evolve(population, fitness_scores)
    
    # Final evaluation
    final_fitness = ga.evaluate_population(population, X_train, y_train, X_test, y_test)
    best_individual_idx = np.argmax(final_fitness)
    best_individual = population[best_individual_idx]
    best_fitness = final_fitness[best_individual_idx]
    
    print("\n=== Final Results ===")
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Best Individual: {best_individual}")
    
    # Visualizations
    plot_fitness_progress(range(1, generations + 1), best_fitness_history, avg_fitness_history)
    plot_feature_selection(best_individual, feature_names)

if __name__ == "__main__":
    main()