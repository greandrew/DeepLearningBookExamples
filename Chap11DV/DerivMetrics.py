import torch

def compute_metrics(result, target):
    # Ensure result and target have the same size
    assert result.size() == target.size(), "Size mismatch between provided tensors."

    # 1. Mean Error
    mean_error = torch.mean(result - target)

    # 2. Mean Absolute Error
    mae = torch.mean(torch.abs(result - target))

    # 3. Mean Squared Error
    mse = torch.mean((result - target) ** 2)

    # 4. Coefficient of Determination
    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - result) ** 2)
    r2 = 1 - (ss_res / ss_total)

    # 5. Median Absolute Error
    medae = torch.median(torch.abs(result - target))

    # 6. Max Error
    max_error = torch.max(torch.abs(result - target))

    return {
        'Mean Error': mean_error.item(),
        'Mean Absolute Error': mae.item(),
        'Mean Squared Error': mse.item(),
        'R^2': r2.item(),
        'Median Absolute Error': medae.item(),
        'Max Error': max_error.item(),
    }

def evaluate_metrics(data_loader, model):
    all_results = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            all_results.append(outputs.squeeze())
            all_targets.append(targets)

    # Concatenate all results and targets
    all_results = torch.cat(all_results)
    all_targets = torch.cat(all_targets)

    # Compute metrics on the entire dataset
    metrics = compute_metrics(all_results, all_targets)

    return metrics, all_results, all_targets