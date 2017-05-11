import theano, theano.tensor as T

# Basically average cosine similarity
# ----
#   Ref: https://arxiv.org/abs/1609.03126
def pullaway_cost(input):
    if input.ndim > 2:
        input = input.flatten(2)

    normal_input = input / T.sqrt(T.sum(input**2, axis=1, keepdims=True))

    cosine_mat = T.dot(normal_input, normal_input.T)

    mask = 1. - T.eye(input.shape[0])

    avg_cosine = T.sum(cosine_mat * mask) / (input.shape[0] * (input.shape[0] - 1.))

    return avg_cosine