import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def fitting(model, name, x_train, y_train):
    model.fit(x_train, y_train)
    print(name + " has been fitted.")
    return model


def pickling(model, folder, name):
    pickle.dump(model, open(folder + name + '.pkl', 'wb'))
    print(name + " has been pickled.")


def fitckling(models, x_train, y_train, folder, names):
    for i in range(len(models)):
        model = fitting(models[i], names[i], x_train, y_train)
        pickling(model, folder, names[i])
