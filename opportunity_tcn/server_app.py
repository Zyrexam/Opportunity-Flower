# server.py

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context
from opportunity_tcn.task import build_models
 


# Optional: Define metric aggregation function
def weighted_average(metrics):
    total_samples = sum(sample_num for sample_num, _ in metrics)
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        weighted_metrics[key] = sum(sample_num * m[key] for sample_num, m in metrics) / total_samples
    return weighted_metrics


def get_parameters(model):
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def server_fn(context: Context) -> ServerAppComponents:
    # Number of rounds (can also be passed via run-config)
    num_rounds = int(context.run_config.get("num-server-rounds", 5))

    # Initialize model to get initial parameters
    encoder, _, classifier = build_models(num_classes=4)  # Set correct number of classes
    model = torch.nn.Sequential(encoder, classifier)
    weights = get_parameters(model)
    parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Register the ServerApp
app = ServerApp(server_fn=server_fn)
