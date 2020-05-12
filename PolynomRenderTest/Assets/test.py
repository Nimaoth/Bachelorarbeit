import pickle
import numpy

pickle_in = open("D:\\Bachelorarbeit\\learned-subsurface-scattering\\pysrc\\outputs_paper\\vae3d\\datasets\\0118_ScatterDataMixed3\\train\\data_stats.pickle", "rb")
X = pickle.load(pickle_in)
print(X)