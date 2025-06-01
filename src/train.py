import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # Example, replace with actual dataset

# Assuming model.py is in the same directory or accessible via PYTHONPATH
from model import SimpleCNN

# A dummy dataset for illustration
def get_dummy_dataloader(batch_size, num_classes, img_size=(1, 28, 28)):
    # Create dummy data: 1000 samples, img_size, random labels
    num_samples = 1000
    data = torch.randn(num_samples, *img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # --- Device Setup ---
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Initialization ---
    # Example: num_classes could come from data module or be fixed
    model = SimpleCNN(num_classes=cfg.model.params.num_classes).to(device)
    print(f"Model: {cfg.model.name}")

    # --- Dataloader ---
    # Replace with your actual dataloader
    train_loader = get_dummy_dataloader(
        batch_size=cfg.training.batch_size,
        num_classes=cfg.model.params.num_classes
    )
    # val_loader = get_dummy_dataloader(cfg.training.batch_size, cfg.model.params.num_classes)


    # --- Optimizer ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.optimizer.lr
    )

    # --- Loss Function ---
    criterion = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % cfg.training.log_interval == 0:
                print(f"Epoch: {epoch+1}/{cfg.training.epochs} | "
                      f"Batch: {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        # --- Validation Step (Optional) ---
        # model.eval()
        # val_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for data, target in val_loader:
        #         data, target = data.to(device), target.to(device)
        #         output = model(data)
        #         val_loss += criterion(output, target).item()
        #         pred = output.argmax(dim=1, keepdim=True)
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        # val_loss /= len(val_loader.dataset)
        # print(f"Epoch: {epoch+1} Validation Avg Loss: {val_loss:.4f}, "
        #       f"Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)")

    print("Training finished.")

    # --- Save Model (Optional) ---
    # Example: Save the trained model
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # model_path = f"{output_dir}/final_model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
