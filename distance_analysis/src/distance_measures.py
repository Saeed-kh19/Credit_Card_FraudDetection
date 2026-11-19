import math


def _check_vectors(point_a, point_b):
    a = list(point_a)
    b = list(point_b)

    if len(a) != len(b):
        raise ValueError("Both points must have the same number of features.")
    if len(a) == 0:
        raise ValueError("Points cannot be empty.")

    return a, b


def euclidean_distance(point_a, point_b):
    a, b = _check_vectors(point_a, point_b)
    total = 0.0

    for i in range(len(a)):
        difference = float(a[i]) - float(b[i])
        total += difference ** 2

    return math.sqrt(total)


def manhattan_distance(point_a, point_b):
    a, b = _check_vectors(point_a, point_b)
    total = 0.0

    for i in range(len(a)):
        total += abs(float(a[i]) - float(b[i]))

    return total


def cosine_distance(point_a, point_b):
    a, b = _check_vectors(point_a, point_b)
    dot_product = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0

    for i in range(len(a)):
        value_a = float(a[i])
        value_b = float(b[i])
        dot_product += value_a * value_b
        norm_a_squared += value_a ** 2
        norm_b_squared += value_b ** 2

    norm_a = math.sqrt(norm_a_squared)
    norm_b = math.sqrt(norm_b_squared)

    if norm_a == 0 and norm_b == 0:
        return 0.0
    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = dot_product / (norm_a * norm_b)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def pairwise_distance_matrix(points, metric="euclidean"):
    distance_functions = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_distance,
    }

    if metric not in distance_functions:
        raise ValueError("metric must be 'euclidean', 'manhattan', or 'cosine'.")

    rows = [list(row) for row in points]
    if not rows:
        raise ValueError("At least one data point is required.")

    feature_count = len(rows[0])
    if feature_count == 0 or any(len(row) != feature_count for row in rows):
        raise ValueError("The input must be a non-empty rectangular feature matrix.")

    distance_function = distance_functions[metric]
    size = len(rows)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(i + 1, size):
            distance = distance_function(rows[i], rows[j])
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix
