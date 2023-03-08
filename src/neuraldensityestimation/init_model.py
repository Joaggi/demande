from generate_model import generate_model


def init_model(setting):
    model = generate_model(setting["z_hidden_shape"], setting["z_n_layers"])

    # initialize flow
    samples = model.sample(sample_shape=(1000,2))
    print(samples)

    print(model.log_prob(samples))
    return model

