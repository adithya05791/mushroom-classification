import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, elitism_count):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
    def initialize_population(self):
        return np.random.randint(2, size=(self.population_size, self.chromosome_length))
    
    def fitness(self, individual, X_train, y_train, X_test, y_test):
        # Get selected features (where gene == 1)
        selected_features = [i for i in range(len(individual)) if individual[i] == 1]
        
        if not selected_features:
            return 0
        
        # Train decision tree on selected features
        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_selected, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Penalize for too many features (optional)
        feature_penalty = len(selected_features) / self.chromosome_length
        fitness_score = accuracy * (1 - 0.1 * feature_penalty)
        
        return fitness_score
    
    def evaluate_population(self, population, X_train, y_train, X_test, y_test):
        return np.array([self.fitness(individual, X_train, y_train, X_test, y_test) 
                         for individual in population])
    
    def selection(self, population, fitness_scores):
        # Tournament selection
        selected_indices = []
        for _ in range(self.population_size - self.elitism_count):
            # Randomly select 3 individuals and pick the best one
            candidates = np.random.choice(len(population), size=3, replace=False)
            best_candidate = candidates[np.argmax(fitness_scores[candidates])]
            selected_indices.append(best_candidate)
        
        return population[selected_indices]
    
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            # Single-point crossover
            crossover_point = random.randint(1, self.chromosome_length - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # Flip the bit
        return individual
    
    def evolve(self, population, fitness_scores):
        new_population = np.empty_like(population)
        
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        new_population[:self.elitism_count] = population[elite_indices]
        
        # Selection
        selected_parents = self.selection(population, fitness_scores)
        
        # Crossover and mutation
        for i in range(self.elitism_count, self.population_size, 2):
            parent1, parent2 = random.choices(selected_parents, k=2)
            child1, child2 = self.crossover(parent1, parent2)
            new_population[i] = self.mutation(child1)
            if i + 1 < self.population_size:
                new_population[i + 1] = self.mutation(child2)
        
        return new_population