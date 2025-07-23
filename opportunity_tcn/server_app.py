import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context
from opportunity_tcn.task import build_models


# Weighted averaging for metrics across clients
def weighted_average(metrics):
    total_samples = sum(sample_num for sample_num, _ in metrics)
    weighted_metrics = {}
    for key in metrics[0][1].keys():
        weighted_metrics[key] = sum(sample_num * m[key] for sample_num, m in metrics) / total_samples
    return weighted_metrics


# Convert PyTorch model state_dict to NumPy parameters for FL
def get_parameters(model):
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


# Server function used by Flower
def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    num_clients = int(context.run_config.get("num-supernodes", 4))
    
    num_classes = int(context.run_config.get("num_classes", 18))
    encoder, _, classifier = build_models(num_classes=num_classes)

    model = torch.nn.Sequential(encoder, classifier)
    parameters = ndarrays_to_parameters(get_parameters(model))

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_available_clients=num_clients,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# Register the server application
app = ServerApp(server_fn=server_fn)
