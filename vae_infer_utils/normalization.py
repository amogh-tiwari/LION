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

    print(f"Scaling params; Mean: {all_points_mean} | Std: {all_points_std}")        
    all_points = (all_points - all_points_mean) / all_points_std
    return all_points, all_points_mean, all_points_std
