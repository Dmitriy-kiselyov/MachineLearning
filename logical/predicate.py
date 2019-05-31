def create_predicate(best_gain):
    feat = best_gain["feat"]
    value_from = best_gain["value"]["from"]
    value_to = best_gain["value"]["to"]

    def predicate(x):
        return value_from <= x[feat] <= value_to

    predicate.to_str = __create_to_str(best_gain)

    return predicate


def __create_to_str(best_gain):
    feat = best_gain["feat"]
    value_from = best_gain["value"]["from"]
    value_to = best_gain["value"]["to"]
    cls = best_gain["class"]

    def to_str():
        if value_from == -float('inf') and value_to == float('inf'):
            return "else ⇒ " + cls

        feat_char = "₀₁₂₃₄₅₆₇₈₉"[feat]

        s = "["
        if value_from != -float('inf'):
            s += str(value_from) + " ≤ "
        s += "f" + feat_char + "(x)"
        if value_to != float('inf'):
            s += " ≤ " + str(value_to)
        s += "] ⇒ " + cls

        return s

    return to_str
