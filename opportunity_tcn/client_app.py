import os
import torch
import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context
from opportunity_tcn.task import load_client_data
from opportunity_tcn.task import build_models, train_classifier, train_simclr, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, data, num_classes, mode: str = "classification"):
        self.data = data
        self.mode = mode
        self.num_classes = num_classes

        # Build models
        self.encoder, self.simclr_model, self.classifier = build_models(num_classes=self.num_classes)
        self.encoder.to(device)
        self.classifier.to(device)
        self.simclr_model.to(device)

        # Optimizer
        if mode == "classification":
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=1e-3
            )
        else:
            self.optimizer = torch.optim.Adam(self.simclr_model.parameters(), lr=1e-3)

    def get_parameters(self, config):
        if self.mode == "classification":
            return [v.cpu().numpy() for v in self.encoder.state_dict().values()] + \
                   [v.cpu().numpy() for v in self.classifier.state_dict().values()]
        else:
            return [v.cpu().numpy() for v in self.simclr_model.state_dict().values()]

    def set_parameters(self, parameters):
        if self.mode == "classification":
            enc_keys = list(self.encoder.state_dict().keys())
            cls_keys = list(self.classifier.state_dict().keys())
            enc_params = parameters[:len(enc_keys)]
            cls_params = parameters[len(enc_keys):]

            enc_state = {k: torch.tensor(v) for k, v in zip(enc_keys, enc_params)}
            cls_state = {k: torch.tensor(v) for k, v in zip(cls_keys, cls_params)}

            self.encoder.load_state_dict(enc_state)
            self.classifier.load_state_dict(cls_state)
        else:
            simclr_keys = list(self.simclr_model.state_dict().keys())
            simclr_state = {k: torch.tensor(v) for k, v in zip(simclr_keys, parameters)}
            self.simclr_model.load_state_dict(simclr_state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config.get("epochs", 1)

        if self.mode == "classification":
            train_classifier(self.encoder, self.classifier, self.data["finetune_loader"], self.optimizer, device, epochs=int(epochs))
        else:
            train_simclr(self.simclr_model, self.data["contrastive_loader"], self.optimizer, device, epochs=int(epochs))

        return self.get_parameters(config), len(self.data["X_train"]), {}

    def evaluate(self, parameters, config):
        print(f"🧪 Evaluating round {config.get('server_round', '?')} | Client {config.get('partition_id', '?')}")
        self.set_parameters(parameters)

        if self.mode == "classification":
            acc, _, _, f1 = evaluate(self.encoder, self.classifier, self.data["test_loader"], device)
            print(f"✅ Accuracy: {acc:.4f}, F1: {f1:.4f}")
            return float(1 - acc), len(self.data["X_test"]), {"accuracy": float(acc), "f1": float(f1)}
        else:
            return 0.0, 0, {"message": "Contrastive mode - evaluation skipped"}

def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    client_folder = os.path.join("OPP_DATA_FLWR", f"Subject_{partition_id + 1}")

    print(f"📦 Loading data for {client_folder}")

    mode = context.run_config.get("mode", "classification")
    batch_size = int(context.run_config.get("batch_size", 64))
    num_classes = int(context.run_config.get("num_classes", 18))

    data = load_client_data(
        client_folder=client_folder,
        initial_window_size=50, W_min=10, W_max=100, shift=10,
        train_ratio=0.1, finetune_ratio=0.9, test_ratio=0.1,
        batch_size=batch_size
    )
    data["num_classes"] = num_classes

    return FederatedClient(data, num_classes=num_classes, mode=str(mode)).to_client()


app = ClientApp(client_fn=client_fn)
