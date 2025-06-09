import matplotlib.pyplot as plt
import seaborn as sns

def plot_fitness_progress(generations, best_fitness, avg_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label='Best Fitness')
    plt.plot(generations, avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Progress Over Generations')
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_selection(best_individual, feature_names):
    selected_features = [feature_names[i] for i in range(len(best_individual)) if best_individual[i] == 1]
    non_selected_features = [feature_names[i] for i in range(len(best_individual)) if best_individual[i] == 0]
    
    print(f"Selected Features ({len(selected_features)}): {selected_features}")
    print(f"Non-Selected Features ({len(non_selected_features)}): {non_selected_features}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=best_individual)
    plt.xticks(rotation=90)
    plt.title('Feature Selection (1 = selected, 0 = not selected)')
    plt.tight_layout()
    plt.show()