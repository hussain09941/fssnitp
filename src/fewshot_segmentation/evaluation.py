"""Fold definitions and model summary utilities."""
def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())


    trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel parameters:")


    print(f"Total     : "f"{total_params:,}")

    print(f"Trainable : "f"{trainable_params:,}")

