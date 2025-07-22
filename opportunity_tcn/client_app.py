# client.py

import os
import torch
import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context
from dataset import load_subject_windows
from opportunity_tcn.task  import (
    build_models, get_dataloaders, split_datasets,
    train_classifier, train_simclr, evaluate
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Flower Federated Client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder, classifier, simclr_model, train_loader, test_loader, mode: str = "classification"):
        self.encoder = encoder
        self.classifier = classifier
        self.simclr_model = simclr_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.mode = mode

        if mode == "classification":
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=1e-3
            )
        else:
            self.optimizer = torch.optim.Adam(self.simclr_model.parameters(), lr=1e-3)

    def get_parameters(self, config):
        if self.mode == "classification":
            return [val.cpu().numpy() for val in self.encoder.state_dict().values()] + \
                   [val.cpu().numpy() for val in self.classifier.state_dict().values()]
        else:
            return [val.cpu().numpy() for val in self.simclr_model.state_dict().values()]

    def set_parameters(self, parameters):
        if self.mode == "classification":
            enc_keys = list(self.encoder.state_dict().keys())
            cls_keys = list(self.classifier.state_dict().keys())
            enc_params = parameters[:len(enc_keys)]
            cls_params = parameters[len(enc_keys):]
            self.encoder.load_state_dict(dict(zip(enc_keys, map(torch.tensor, enc_params))), strict=True)
            self.classifier.load_state_dict(dict(zip(cls_keys, map(torch.tensor, cls_params))), strict=True)
        else:
            simclr_keys = list(self.simclr_model.state_dict().keys())
            self.simclr_model.load_state_dict(dict(zip(simclr_keys, map(torch.tensor, parameters))), strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.mode == "classification":
            train_classifier(self.encoder, self.classifier, self.train_loader, self.optimizer, device=device)
        else:
            train_simclr(self.simclr_model, self.train_loader, self.optimizer, device=device)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.mode == "classification":
            acc, _, _, f1 = evaluate(self.encoder, self.classifier, self.test_loader, device=device)
            return float(1 - acc), len(self.test_loader.dataset), {"accuracy": float(acc), "f1": float(f1)}
        else:
            return 0.0, 0, {"message": "Contrastive mode - evaluation skipped"}

# Flower ClientApp function
def client_fn(context: Context) -> fl.client.Client:
    # Get client ID and training mode
    partition_id = context.node_config["partition_id"]
    mode = context.run_config.get("mode", "classification")

    # Load subject data
    subject_folder = os.path.join("OpportunityData", f"subject{int(partition_id) + 1}")
    windows, labels = load_subject_windows(subject_folder,
                                           initial_window_size=50,
                                           W_min=10, W_max=100,
                                           shift=10)

    # Split and prepare dataloaders
    X_train, y_train, X_ft, y_ft, X_test, y_test = split_datasets(windows, labels)
    contrastive_loader, classifier_loader, test_loader = get_dataloaders(
        X_train, y_train, X_ft, y_ft, X_test, y_test
    )

    # Build model components
    encoder, simclr_model, classifier = build_models(num_classes=len(set(labels)))
    encoder.to(device)
    simclr_model.to(device)
    classifier.to(device)

    # Return correct client with .to_client()
    if mode == "classification":
        return FederatedClient(
            encoder, classifier, simclr_model, classifier_loader, test_loader, mode="classification"
        ).to_client()
    else:
        return FederatedClient(
            encoder, classifier, simclr_model, contrastive_loader, test_loader, mode="contrastive"
        ).to_client()


# Register ClientApp
app = ClientApp(client_fn=client_fn)
