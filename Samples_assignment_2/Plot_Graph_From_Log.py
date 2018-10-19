import matplotlib.pyplot as plt

def plot_graph(file):
    with open(file) as f:
        data = f.read()
        data = data.split(" ")
        data = data[0:10001]
        values = []
        for d in data:
            value = float(d)
            values.append(value)


        plt.figure('Genetic Algorithm')
        plt.plot(range(len(values)), values, 'r', label='Fitness') # r=red color, b=blue color
        plt.xlabel('Iterations')
        plt.ylabel('Fitness value')
        plt.title('Genetic Algorithm')
        plt.show()




plot_graph("log1.txt")

