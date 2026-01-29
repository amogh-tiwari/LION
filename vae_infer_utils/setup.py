# helper.py needs these imports at the top:
import importlib
import torch
from loguru import logger
from utils import utils  # or however you import it

def setup(args, config):
    # -- trainer -- #
    logger.info('use trainer: {}', config.trainer.type)
    trainer_lib = importlib.import_module(config.trainer.type)
    Trainer = trainer_lib.Trainer

    if config.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        logger.info(
            '\n\n' + '!'*30 + '\nWARNING: ths set_detect_anomaly is turned on, it can slow down the training! \n' + '!'*30)

    # -- command init -- #
    comet_key = config.comet_key
    _, writer = utils.common_init(args.global_rank,
                                  config.trainer.seed, config.save_dir, comet_key)
    trainer = Trainer(config, args)
    nparam = utils.count_parameters_in_M(trainer.model)
    logger.info('param size = %fM ' % nparam)

    # -- Load checkpoint (this part stays from original) -- #
    if args.resume or args.eval_generation:
        if args.pretrained is not None:
            trainer.start_epoch = trainer.resume(
                args.pretrained, eval_generation=args.eval_generation)
        else:
            raise NotImplementedError
    elif args.pretrained is not None:
        logger.info('Loading pretrained weights from: {}', args.pretrained)
        trainer.resume(args.pretrained)

    logger.info('='*80)
    logger.info('VAE MODEL LOADED SUCCESSFULLY!')
    logger.info('='*80)
    logger.info('Model parameters: {:.2f}M', nparam)
    
    return args, config, trainer, writer, nparam

# def run_on_dir(trainer, in_dir, out_dir, normalization_type, batch_size=4):
#     file_names = os.listdir(in_dir)
#     tgt_file_exts = ["ply", "obj"]
#     file_names = [f for f in file_names 
#                 if not os.path.isdir(f) and f.split(".")[-1] in tgt_file_exts]
#     file_names = [f for f in file_names if not "body.ply" in f]
#     all_points = []
#     for fn in file_names:
#         fp = os.path.join(in_dir, fn)
#         verts = trimesh.load(fp).vertices
#         verts_sampled = sample_vertices(verts, NUM_VERTICES_TO_SAMPLE)
#         all_points.append(verts_sampled)
#     all_points = np.asarray(all_points)
#     all_points = torch.tensor(all_points, device='cuda', dtype=torch.float32)
#     print(f"Shape: {all_points.shape} | Mean: {all_points.mean()} | Max: {all_points.max()} | Min: {all_points.min()}")
#     all_points_norm, all_points_mean, all_points_std = normalize(all_points, normalization_type)
#     print(f"Shape: {all_points_norm.shape} | Mean: {all_points_norm.mean()} | Max: {all_points_norm.max()} | Min: {all_points_norm.min()}")
#     print(all_points.shape)

#     all_outputs = []
#     for i in range(0, len(all_points_norm), batch_size):
#         batch = all_points_norm[i:i+batch_size]
#         out = trainer.model.recont(batch)
#         out_sample = {}
#         # Assuming batch size is always 1
#         out_sample = {
#             'mu': out['latent_list'][0][1][0].detach().cpu().numpy(),
#             'sigma': out['latent_list'][0][2][0].detach().cpu().numpy(),
#         }

#         os.makedirs(out_dir, exist_ok=True)
#         os.makedirs(os.path.join(out_dir, "latent_codes"), exist_ok=True)
#         os.makedirs(os.path.join(out_dir, "embeds"), exist_ok=True)

#         tgt_obj_name = file_names[i].split('.')[0]
#         np.savez_compressed(os.path.join(out_dir, "embeds", f"{tgt_obj_name}"), **out_sample)

#         # Assuming batch size is always 1
#         _ = trimesh.PointCloud(batch[0].detach().cpu().numpy()).export(f"{out_dir}/{tgt_obj_name}_input.obj")
#         _ = trimesh.PointCloud(out['x_0_pred'][0].detach().cpu().numpy()).export(f"{out_dir}/{tgt_obj_name}_recon.obj")
#         _ = trimesh.PointCloud(out['vis/latent_pts'][0].detach().cpu().numpy()).export(f"{out_dir}/latent_pts/{tgt_obj_name}_latent_pts.obj")
    
