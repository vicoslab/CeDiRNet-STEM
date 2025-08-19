import copy
import os
import numpy as np

import torch
from torchvision.transforms import InterpolationMode

NANOPARTICLES_KI_DATASET = os.environ.get('NANOPARTICLES_KI_DATASET_DIR')

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

NUM_FIELDS = 3 + 2 # 3 == cedirnet default,  2 == radius+circularity

SIZE = int(os.environ.get('TRAIN_SIZE', default=512))
SIZE_WIDTH = int(os.environ.get('TRAIN_SIZE_WIDTH', default=SIZE))
SIZE_HEIGHT = int(os.environ.get('TRAIN_SIZE_HEIGHT', default=SIZE))

IN_CHANNELS = 3

args = dict(

	cuda=True,
	cuda_sync_with_file=False,
	display=False,
	display_it=20,


	tf_logging=['loss'],
	tf_logging_iter=2,

	visualizer=dict(name='CentersShapeVisualizeTrain',
				 	opts=dict(shape_type='circle_with_circularity')),

	save=True,
	save_interval=10,

	# --------
	n_epochs=100,
	
	save_dir=os.path.join(OUTPUT_DIR),

	pretrained_model_path = None,
	resume_path = None,

	pretrained_center_name = None,
	pretrained_center_model_path = None,

	train_dataset = {
		'name': 'nanoparticles',
		'kwargs': {
			'root_dir': os.path.abspath(NANOPARTICLES_KI_DATASET),
			'name_pattern': '*_BF.png',
			'name_pattern_for_channels': [("_BF.png",'_HAADF.png'),],
			'gt_name_replace_args': [('_BF.png','_masks.npz'),],
			'subfolders': ['PtNi', 'PtCo','PtCo-deg'],
			'valid_sample_names': [
				# a subset from PtNi
				'exsitu_PtNi_-0021','exsitu_PtNi_-0031','exsitu_PtNi_-0035','exsitu_PtNi_-0039','exsitu_PtNi_-0043','exsitu_PtNi_-0045','exsitu_PtNi_-0075','exsitu_PtNi_-0077','exsitu_PtNi_-0089','exsitu_PtNi_-0091', 
				# a subset from PtCo
				'PtCo_IL_a-0016','PtCo_IL_a-0090','PtCo_IL_a-0118','PtCo_IL_a-0152','PtCo_IL_b-0014','PtCo_IL_b-0093','PtCo_IL_b-0119','PtCo_IL_b-0151', 
				# a subset from PtCo-deg with different aglomerations
				'FCS-1-0053','FCS-1-0139','FCS-1-0161','FCS-1-0261','FCS-1-0279',
			],
			'BORDER_MARGIN_FOR_MASK': 0,
			'fixed_bbox_size': 15,
			'gt_from_circles_fitting': False, # MUST BE SET to False for circularity
			'gt_from_circularity': True,
			'resize_factor': 1,
			
			'transform_per_sample_rng': False,
			'transform': [
				# for training without augmentation (same as testing)
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'mask','shape_coef'),
						'type': (
						torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor),
					}
				},
				{
					'name': 'Resize',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'mask', 'shape_coef'),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
						'keys_bbox': ('center',),
						'keys_custom_fn': dict(shape_coef=lambda I, old_size, new_size: torch.stack((I[0]*(np.mean(np.array(new_size)/np.array(old_size))),I[1]),dim=0)),
						'size': (SIZE_HEIGHT, SIZE_WIDTH),
					}
				},
				{
					'name': 'RandomHorizontalFlip',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore','shape_coef'), 'keys_bbox': ('center',),
						'p': 0.5,
					}
				},
				{
					'name': 'RandomVerticalFlip',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore','shape_coef'), 'keys_bbox': ('center',),
						'p': 0.5,
					}
				},				
				{
					'name': 'ColorJitter',
					'opts': {
						'keys': ('image',), 'p': 0.5,
						'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
					}
				}

			],
			'MAX_NUM_CENTERS':2*1024,
		},

		'centerdir_gt_opts': dict(
			generic_regression_maps=[('shape_coef','gt_shape_coef')],

			# we always use this to generate center-direction for all bg (i.e., weakly-supervised mode)
			ignore_instance_mask_and_use_closest_center=True, 

			# this is neede to ignore 3px around center in loss due to discontinutiy
			center_ignore_px=3,
			# gt_center_mask is not needed since we are not training localization network
			skip_gt_center_mask_generate=True, 

			MAX_NUM_CENTERS=2*1024, 
		),

		'batch_size': 4,
		'hard_samples_size': 0,
		'workers': 4,
		'shuffle': True,
	}, 

	model = dict(
		name='fpn',
		kwargs= {
			'backbone': 'tu-convnext_base',
			'num_classes': [NUM_FIELDS, 1],
			'use_custom_fpn':True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,
			'fpn_args': {
				'upsampling':4, # for 'tu-convnext_*'
				'decoder_segmentation_head_channels':64,
				'classes_grouping': [(0, 1, 2, 5),(3,),(4,)],
			},
			'init_decoder_gain': 0.1
		},
		optimizer='Adam',
		lr=1e-4,
		weight_decay=0,

	),
	center_model=dict(
		name='CenterAttributeEstimator',
		kwargs=dict(
			# define which attributes we regress: key is output name, values are input indexes from predicted maps (begining from the last center direction map)
			# key = shape_coef (as expected for CenterShapeEval)
			attributes=dict(shape_coef=dict(input_start=0, # start index for radius+circulairty
								   			input_end=2, # end index for radius+circulairty
                                   			use_log=True, 
                                            use_log_channels=[0] # apply only to radius (first shape channel)
							)),			

			use_centerdir_radii = False,						
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True, local_max_thr=0.01,
			### dilated neural net as head for center detection
			ignore_centerdir_magnitude=True, 
			ignore_cls_prediction=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
				),
			augmentation=False,
		),
		optimizer='Adam',
		lr=0,
		weight_decay=0,
	),	

	# loss options
	loss_type='ShapeLoss',
	loss_opts={
 		'num_vector_fields': NUM_FIELDS,
        'foreground_weight': 1, 
        
        'no_instance_loss': True, 	# apply loss to all pixels not just around instances (bg and fg)
        'cls_no_loss': True, 		# no classification loss

        # weighting should be only inside mask (we use small fixed area around center!)
        'cls_instance_weighted': True,
        'centerdir_instance_weighted': True,
		
        'cls_loss': 'l1',			# regressing using L1 loss
        'regression_loss': 'l1',	# regressing using L1 loss

        'learnable_center_est': False, # do not learn centers (need to use pretrained center model)

        'enable_centerdir_loss': True,	# loss for center-directions
        'enable_cls_loss': False,		# no loss for classificaiton (disabled)
		
		'shape_args': dict(
			enable=True,
			shape_type='circle_with_circularity',
			no_instance_loss=False, 	# we learn orientation only inside mask
			regression_loss='l1',		# regressing using L1 loss
			
			individual_uw=False,

			use_log = True,
			use_log_channels=[0], 		# add log only to radius (first channel)
		)
	},
	multitask_weighting=dict(
		name='uw',
		kwargs=dict(
			n_tasks=2
		)
	),
	loss_w={
		'w_cent': 0.1,
		'w_shape': 0.5,
		'w_radius': 1,
		'w_circularity': 1,
	},

)

# Original scheduler used by SpatialEmbedding method
args['lambda_scheduler_fn']=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9))
#args['lambda_scheduler_fn']=lambda _args: (lambda epoch: 1.0) # disabled

args['model']['lambda_scheduler_fn'] = args['lambda_scheduler_fn']
args['center_model']['lambda_scheduler_fn'] = lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9) if epoch > 1 else 0)


def get_args():
	return copy.deepcopy(args)
