import os


s = "-"
seq = ("a", "b", "c")
print (s.join(seq))


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)
print(os.getcwd())