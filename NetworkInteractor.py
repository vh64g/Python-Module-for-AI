import pickle
import random

from matplotlib import pyplot as plt

from ai import neuralNetwork

network = None
data = []

def create_data(amount=100, _min=0, _max=1):
    global data
    data = []
    for i in range(amount):
        data.append(
            [
                [x := random.random(), y := random.random()],
                [1 if ((x + y) / 2) > 0.5 else 0]
            ])

def calc(calc_data):
    if network is None:
        print("No network loaded")
        return
    fig, ax = plt.subplots()
    for _ in calc_data:
        print(f"{round((calc_data.index(_) / len(calc_data)) * 100)}%")
        z = network.calc(_[0])
        if z[0] < 0.5: ax.scatter(_[0][0], _[0][1], c="g")
        else: ax.scatter(_[0][0], _[0][1], c="y")
    plt.show()

def main():
    global network
    global data
    while True:
        cmd = input(">>> ")
        if cmd == "exit": break
        if cmd == "rn": network = pickle.load(open("network.txt", "rb"))
        if cmd == "cd": create_data(amount=int(input("Amount: ")), _min=int(input("Min: ")), _max=int(input("Max: ")))
        if cmd == "pd": print(data)
        if cmd == "pn": print(network)
        if cmd == "c": calc(data)
        if cmd == "ctr": pass

if __name__ == "__main__":
    main()