import random
import json
import prettytable

# Constants
NUM_LESSONS = 6
NUM_TEACHERS = 3
NUM_CLASSES = 2
NUM_DAYS = 5
MAX_LESSONS_PER_DAY = 5

# Load data from file
with open('data.json', 'r') as f:
    data = json.load(f)
    lessons = data['lessons']
    teachers = data['teachers']
    classes = data['classes']
    rooms = data['rooms']

# Define the schedule class
class Schedule:
    def __init__(self):
        self.schedule = [[[None for _ in range(MAX_LESSONS_PER_DAY)] for _ in range(NUM_DAYS)] for _ in range(NUM_CLASSES)]

    def __str__(self):
        result = ''
        for c in range(NUM_CLASSES):
            result += f'\n{classes[c]}:\n'
            table = prettytable.PrettyTable()
            table.hrules = prettytable.ALL
            table.field_names = ["Day", "Lesson 1", "Lesson 2", "Lesson 3", "Lesson 4", "Lesson 5"]
            for d in range(NUM_DAYS):
                row = [f'Day {d+1}']
                for p in range(MAX_LESSONS_PER_DAY):
                    lesson = self.schedule[c][d][p]
                    if lesson is not None:
                        teacher = lessons[lesson]['teacher']
                        row.append(f'{lesson}\n{teacher}')
                    else:
                        row.append('')
                table.add_row(row)
            result += str(table) + '\n'
        return result

    def fitness(self):
        score = 0

        # Check if all lessons are scheduled
        lesson_count = {}
        for c in range(NUM_CLASSES):
            for d in range(NUM_DAYS):
                for p in range(MAX_LESSONS_PER_DAY):
                    lesson = self.schedule[c][d][p]
                    if lesson is not None:
                        if lesson not in lesson_count:
                            lesson_count[lesson] = 0
                        lesson_count[lesson] += 1

        for lesson in lessons:
            if lesson in lesson_count and lesson_count[lesson] == lessons[lesson]['count']:
                score += 1

        # Check if teachers are not double-booked
        teacher_count = {}
        for c in range(NUM_CLASSES):
            for d in range(NUM_DAYS):
                for p in range(MAX_LESSONS_PER_DAY):
                    lesson = self.schedule[c][d][p]
                    if lesson is not None:
                        teacher = lessons[lesson]['teacher']
                        if (teacher, d, p) not in teacher_count:
                            teacher_count[(teacher, d, p)] = 0
                        teacher_count[(teacher, d, p)] += 1

        for count in teacher_count.values():
            if count == 1:
                score += 1

        # Check if specialized rooms are not double-booked
        room_count = {}
        for c in range(NUM_CLASSES):
            for d in range(NUM_DAYS):
                for p in range(MAX_LESSONS_PER_DAY):
                    lesson = self.schedule[c][d][p]
                    if lesson is not None and 'room' in lessons[lesson]:
                        room = lessons[lesson]['room']
                        if (room, d, p) not in room_count:
                            room_count[(room, d, p)] = 0
                        room_count[(room, d, p)] += 1

        for count in room_count.values():
            if count == 1:
                score += 1

        return score

    def crossover(self, other):
        child1 = Schedule()
        child2 = Schedule()

        # Crossover days
        for c in range(NUM_CLASSES):
            day_split = random.randint(0, NUM_DAYS - 1)
            for d in range(day_split):
                child1.schedule[c][d] = self.schedule[c][d][:]
                child2.schedule[c][d] = other.schedule[c][d][:]
            for d in range(day_split, NUM_DAYS):
                child1.schedule[c][d] = other.schedule[c][d][:]
                child2.schedule[c][d] = self.schedule[c][d][:]

        return child1, child2

    def mutate(self):
        # Mutate by swapping two periods within a day
        c = random.randint(0, NUM_CLASSES - 1)
        d = random.randint(0, NUM_DAYS - 1)
        p1 = random.randint(0, MAX_LESSONS_PER_DAY - 1)
        p2 = random.randint(0, MAX_LESSONS_PER_DAY - 1)
        self.schedule[c][d][p1], self.schedule[c][d][p2] = self.schedule[c][d][p2], self.schedule[c][d][p1]

# Genetic algorithm
def genetic_algorithm(population_size=100, num_generations=1000, mutation_rate=0.01):
    # Initialize population
    population = [Schedule() for _ in range(population_size)]
    for schedule in population:
        # Randomly assign lessons to periods
        for c in range(NUM_CLASSES):
            periods = []
            for d in range(NUM_DAYS):
                periods.extend([(c, d, p) for p in range(MAX_LESSONS_PER_DAY)])
            random.shuffle(periods)

            i = 0
            for lesson, info in lessons.items():
                count = info['count']
                while count > 0 and i < len(periods):
                    period = periods[i]
                    schedule.schedule[period[0]][period[1]][period[2]] = lesson
                    i += 1
                    count -= 1

    # Run genetic algorithm
    best_schedule = None
    best_fitness = -1
    for generation in range(num_generations):
        # Calculate fitness of population
        fitnesses = [schedule.fitness() for schedule in population]

        # Find best schedule
        max_fitness_index = fitnesses.index(max(fitnesses))
        max_fitness_schedule = population[max_fitness_index]
        max_fitness_value = fitnesses[max_fitness_index]

        if max_fitness_value > best_fitness:
            best_schedule = max_fitness_schedule
            best_fitness = max_fitness_value

            # Selection
            new_population = []
            for _ in range(population_size // 2):
                # Select two parents using tournament selection
                parent1 = population[random.randint(0, population_size - 1)]
                parent2 = population[random.randint(0, population_size - 1)]
                if parent1.fitness() > parent2.fitness():
                    new_population.append(parent1)
                else:
                    new_population.append(parent2)

                parent1 = population[random.randint(0, population_size - 1)]
                parent2 = population[random.randint(0, population_size - 1)]
                if parent1.fitness() > parent2.fitness():
                    new_population.append(parent1)
                else:
                    new_population.append(parent2)

            # Crossover
            population = []
            for i in range(0, population_size, 2):
                child1, child2 = new_population[i].crossover(new_population[i + 1])
                population.append(child1)
                population.append(child2)

            # Mutation
            for schedule in population:
                if random.random() < mutation_rate:
                    schedule.mutate()

    return best_schedule

# Run genetic algorithm
best_schedule = genetic_algorithm()
print(best_schedule)
