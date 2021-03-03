def gen_shift_func(shift_value, vertical=True):
    axis = 1 if vertical else 0

    def shift(xy):
        xy[:, axis] += shift_value
        return xy

    return shift