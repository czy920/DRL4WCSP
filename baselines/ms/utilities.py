def elementwise_add(list1, list2):
    assert len(list1) == len(list2)
    return [x + y for x, y in zip(list1, list2)]


def argmin(arr):
    best_val = arr[0]
    best_idx = 0
    for idx, val in enumerate(arr):
        if val < best_val:
            best_val = val
            best_idx = idx
    return best_idx