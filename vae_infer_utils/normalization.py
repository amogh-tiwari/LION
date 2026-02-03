import torch

def _global_normalize_given_params(all_points, all_points_mean, all_points_std):
    print(f"Scaling params; Mean: {all_points_mean} | Std: {all_points_std}")        
    all_points = (all_points - all_points_mean) / all_points_std
    return all_points, all_points_mean, all_points_std

def normalize(all_points):
    input_dim = 3
    normalize_std_per_axis = False

    all_points_mean = all_points.reshape(
    -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)

    if normalize_std_per_axis:
        # Separate std for each dimension (x, y, z)
        all_points_std = all_points.reshape(
            -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
    else:
        # Single std across all dimensions
        all_points_std = all_points.reshape(-1).std(
            axis=0).reshape(1, 1, 1)

    all_points, all_points_mean, all_points_std = _global_normalize_given_params(all_points, all_points_mean, all_points_std)
    return all_points, all_points_mean, all_points_std

def normalize_ho3d(all_points, normalization_type):
    normalization_type == "global"
    device = all_points.device
    ho3d_normalization_params = {
        'all_points_mean': torch.tensor([[[-0.00059158 -0.00091937 -0.00367902]]]).to(device=device, dtype=torch.float32),
        'all_points_std': torch.tensor([[[0.03672426]]]).to(device=device, dtype=torch.float32),
    }

    all_points, all_points_mean, all_points_std = _global_normalize_given_params(all_points, ho3d_normalization_params['all_points_mean'], ho3d_normalization_params['all_points_std'])
    return all_points, all_points_mean, all_points_std
