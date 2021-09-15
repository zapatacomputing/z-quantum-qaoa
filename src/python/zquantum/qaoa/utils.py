import numpy as np
import random

# For max cut / max k-sat, there are two ways to represent a cut / a variable assignment:
# e.g. x = [0, 1, 1, 0, 0, 1]
#      z = [1, -1, -1, 1, 1, -1]
# We often need to enumerate such strings and convert one representation to another
def get_x_vec(num, n=0):
    b = format(num, 'b').zfill(n)[::-1]
    return np.array([int(a) for a in b])

def get_z_vec(num, n=0):
    b = format(num, 'b').zfill(n)[::-1]
    return np.array([(-1)**(int(a)) for a in b])

def x_vec_to_z_vec(x):
    return np.array([(-1)**(int(a)) for a in x])

def z_vec_to_x_vec(z):
    return np.array([int((1-a)/2) for a in z])

# For hyperplane rounding and its variants, we need to generate random (unit) vectors in high-dimensional space
def generate_random_vector(dim):
    vector = np.random.normal(0.0, 1.0, dim)
    return vector

def generate_random_vectors(dim, num):
    vectors = []
    for _ in range(num):
        vectors.append(generate_random_vector(dim))
    return vectors

def generate_random_unit_vector(dim):
    vector = np.random.normal(0.0, 1.0, dim)
    vector = vector / np.linalg.norm(vector)
    return vector

def generate_random_unit_vectors(dim, num):
    vectors = []
    for _ in range(num):
        vectors.append(generate_random_unit_vector(dim))
    return vectors

def hyperplane_rounding(vectors, last_entry=None):
    dim = vectors.shape[1]
    v = generate_random_unit_vector(dim)
    res = np.array([1 if np.dot(vectors[i], v)>=0 else -1 for i in range(vectors.shape[0])])
    if last_entry is not None and res[-1] != last_entry:
        res *= -1
    return res


def get_x_str(num, n=0):
    return format(num, 'b').zfill(n)[::-1]


def sigmoid(a, b, x):
    return a / (1.0 + np.exp(-b*x))


