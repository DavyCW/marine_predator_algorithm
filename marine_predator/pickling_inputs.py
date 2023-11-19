import pickle


def pickling(data, names):
    for i in range(len(data)):
        data[i].to_pickle("data pickles/" + names[i])
