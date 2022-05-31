def transpose(matrix):
    data = []
    for col in range(len(matrix[0])):
        data.append([matrix[i][col] for i in range(len(matrix))])
    return data