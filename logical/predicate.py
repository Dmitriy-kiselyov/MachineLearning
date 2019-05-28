def create_predicate(best_gain):
    def predicate(x):
        feat = best_gain["feat"]
        x = x[feat]
        return best_gain["value"]["from"] <= x <= best_gain["value"]["to"]

    return predicate
