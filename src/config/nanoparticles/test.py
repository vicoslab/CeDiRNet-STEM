import copy
import os
import numpy as np

import torch
from torchvision.transforms import InterpolationMode

NANOPARTICLES_KI_DATASET = os.environ.get('NANOPARTICLES_KI_DATASET_DIR')

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')
STORAGE_DIR=os.environ.get('STORAGE_DIR',default='/storage')

NUM_FIELDS = 3 + 2
TRAIN_SIZE = int(os.environ.get('TRAIN_SIZE', default=512))
TRAIN_SIZE_WIDTH = int(os.environ.get('TRAIN_SIZE_WIDTH', default=TRAIN_SIZE))
TRAIN_SIZE_HEIGHT = int(os.environ.get('TRAIN_SIZE_HEIGHT', default=TRAIN_SIZE))

SIZE = int(os.environ.get('TEST_SIZE', default=512))
TEST_SIZE_WIDTH = int(os.environ.get('TEST_SIZE_WIDTH', default=SIZE))
TEST_SIZE_HEIGHT = int(os.environ.get('TEST_SIZE_HEIGHT', default=SIZE))

IN_CHANNELS = 3

BACKBONE = 'tu-convnext_base'


impath2name_fn=lambda x: ".".join(x.split('/')[-2:]).replace('.png','').replace('.tif.tiff','')

def to_nanometer_fn(filename):
	filename_to_pixel_size = {
		"PtCo_IL_a-0016": 0.036476,
		"PtCo_IL_a-0018": 0.030397,
		"PtCo_IL_a-0020": 0.060794,
		"PtCo_IL_a-0022": 0.060794,
		"PtCo_IL_a-0090": 0.045596,
		"PtCo_IL_a-0118": 0.036476,
		"PtCo_IL_a-0152": 0.045596,
		"PtCo_IL_a-0160": 0.060794,
		"PtCo_IL_a-0162": 0.022798,
		"PtCo_IL_a-0164": 0.030397,
		"PtCo_IL_b-0012": 0.030397,
		"PtCo_IL_b-0014": 0.036476,
		"PtCo_IL_b-0022": 0.060794,
		"PtCo_IL_b-0024": 0.060794,
		"PtCo_IL_b-0093": 0.045596,
		"PtCo_IL_b-0119": 0.036476,
		"PtCo_IL_b-0151": 0.045596,
		"PtCo_IL_b-0169": 0.060794,
		"PtCo_IL_b-0171": 0.022798,
		"PtCo_IL_b-0173": 0.030397,
		"exsitu_PtNi_-0021": 0.030397,
		"exsitu_PtNi_-0031": 0.036476,
		"exsitu_PtNi_-0035": 0.036476,
		"exsitu_PtNi_-0039": 0.036476,
		"exsitu_PtNi_-0041": 0.036476,
		"exsitu_PtNi_-0043": 0.030397,
		"exsitu_PtNi_-0045": 0.036476,
		"exsitu_PtNi_-0075": 0.030397,
		"exsitu_PtNi_-0077": 0.030397,
		"exsitu_PtNi_-0081": 0.030397,
		"exsitu_PtNi_-0089": 0.036476,
		"exsitu_PtNi_-0091": 0.030397,
	}

	# add for PtCo-deg
	filename_to_pixel_size.update({n: 0.036476 for n in ['FCS-1-0037','FCS-1-0041','FCS-1-0063','FCS-1-0069','FCS-1-0075','FCS-1-0085','FCS-1-0087','FCS-1-0101','FCS-1-0119','FCS-1-0123','FCS-1-0135','FCS-1-0153','FCS-1-0177','FCS-1-0185','FCS-1-0195','FCS-1-0205','FCS-1-0219','FCS-1-0227','FCS-1-0235','FCS-1-0245','FCS-1-0257','FCS-1-0267','FCS-1-0165','FCS-1-0490','FCS-1-0496','FCS-1-0484','FCS-1-0470','FCS-1-0476','FCS-1-0464','FCS-1-0458','FCS-1-0454','FCS-1-0446','FCS-1-0432','FCS-1-0438','FCS-1-0422','FCS-1-0416','FCS-1-0408','FCS-1-0398','FCS-1-0388','FCS-1-0380','FCS-1-0372','FCS-1-0366','FCS-1-0358','FCS-1-0350','FCS-1-0342','FCS-1-0336','FCS-1-0320','FCS-1-0316','FCS-1-0310','FCS-1-0304','FCS-1-0402','FCS-1-0418','FCS-1-0053','FCS-1-0139','FCS-1-0161','FCS-1-0261','FCS-1-0279']})

	px = [px_size for f,px_size in filename_to_pixel_size.items() if f in filename]
	return px[0] if len(px) >= 1 else None


args = dict(

	cuda=True,
	display=False,
	autoadjust_figure_size=True,
	# display to file only
	display_to_file_only=True,
	groundtruth_loading = True,

	save=True,
	save_dir=os.path.join(OUTPUT_DIR,'{args[dataset][kwargs][type]}_results{args[eval_epoch]}'),
	checkpoint_path=os.path.join(OUTPUT_DIR,'checkpoint{args[eval_epoch]}.pth'),

	eval_epoch='',

	eval=dict(
		# available score types ['center']
		score_combination_and_thr=[
			{'center': [0.1,0.01,0.05,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.9,0.94,0.99]},
		],
		score_thr_final=[0.01],
		skip_center_eval=True,
		center_with_shape=dict(
			shape_type='circle_with_circularity',
			display_best_threshold=False,
			tau_thr=[20],
		),
	),
	visualizer=dict(name='CentersShapeVisualizeTest',
					opts=dict(impath2name_fn=impath2name_fn,
			   				  shape_type='circle_with_circularity',
							  # visualize on HAADF images only as grayscale image
							  image_channels=[1],
							  image_grayscale=True)),

	export_detections=dict(name='CenterExport',
                        	opts=dict(impath2name_fn=impath2name_fn,
								   	  attributes=dict(shape_coef=['radii','circularity']),
									  # radii must be multipled by the resize factor (relative to original size), but circularity should not be changed
									  attribute_resize_fn=dict(shape_coef=lambda x,s_x,s_y: np.concatenate([x[:,0:1] * s_x, x[:,1:]],axis=1)))), 

	dataset={
		'name': 'nanoparticles',
		'kwargs': {
			'root_dir': os.path.abspath(NANOPARTICLES_KI_DATASET),
			'name_pattern': '*_BF.png',
			'name_pattern_for_channels': [("_BF.png",'_HAADF.png'),],
			'gt_name_replace_args': [('_BF.png','_masks.npz'),],
			'subfolders': ['PtNi', 'PtCo','PtCo-deg'],
			
			'type': 'test',
			'valid_sample_names': [
				# test samples from PtNi
				'exsitu_PtNi_-0041', 'exsitu_PtNi_-0081',
				# test samples from PtCo
				'PtCo_IL_a-0018','PtCo_IL_a-0020','PtCo_IL_a-0022','PtCo_IL_a-0160','PtCo_IL_a-0162','PtCo_IL_a-0164','PtCo_IL_b-0012','PtCo_IL_b-0022','PtCo_IL_b-0024','PtCo_IL_b-0169','PtCo_IL_b-0171','PtCo_IL_b-0173',
				# test samples from PtCo-deg
				'FCS-1-0037','FCS-1-0041','FCS-1-0063','FCS-1-0069','FCS-1-0075','FCS-1-0085','FCS-1-0087','FCS-1-0101','FCS-1-0119','FCS-1-0123','FCS-1-0135','FCS-1-0153','FCS-1-0177','FCS-1-0185','FCS-1-0195','FCS-1-0205','FCS-1-0219','FCS-1-0227','FCS-1-0235','FCS-1-0245','FCS-1-0257','FCS-1-0267','FCS-1-0165','FCS-1-0490','FCS-1-0496','FCS-1-0484','FCS-1-0470','FCS-1-0476','FCS-1-0464','FCS-1-0458','FCS-1-0454','FCS-1-0446','FCS-1-0432','FCS-1-0438','FCS-1-0422','FCS-1-0416','FCS-1-0408','FCS-1-0398','FCS-1-0388','FCS-1-0380','FCS-1-0372','FCS-1-0366','FCS-1-0358','FCS-1-0350','FCS-1-0342','FCS-1-0336','FCS-1-0320','FCS-1-0316','FCS-1-0310','FCS-1-0304','FCS-1-0402','FCS-1-0418',
			],
			#'type': 'train',
			#'valid_sample_names': [ 'exsitu_PtNi_-0021','exsitu_PtNi_-0031','exsitu_PtNi_-0035','exsitu_PtNi_-0039','exsitu_PtNi_-0043','exsitu_PtNi_-0045','exsitu_PtNi_-0075','exsitu_PtNi_-0077','exsitu_PtNi_-0089','exsitu_PtNi_-0091',  'PtCo_IL_a-0016','PtCo_IL_a-0090','PtCo_IL_a-0118','PtCo_IL_a-0152','PtCo_IL_b-0014','PtCo_IL_b-0093','PtCo_IL_b-0119','PtCo_IL_b-0151',  'FCS-1-0054','FCS-1-0140','FCS-1-0162','FCS-1-0262','FCS-1-0280', ],
			'gt_from_circles_fitting': False, # MUST BE SET to False for circularity
			'gt_from_circularity': True,
			'mapping_to_nm': to_nanometer_fn,
			'BORDER_MARGIN_FOR_MASK': 0,
			'fixed_bbox_size': 5,
			'resize_factor': 1,
			'transform': [
				{
					'name': 'ToTensor',
					'opts': {

						'keys': ('image', 'instance', 'label', 'ignore','mask', 'shape_coef') ,
						'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor),
					}
				},
				{
					'name': 'Resize',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore',  'mask', 'shape_coef'),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
						'keys_bbox': ('center',), # 'instance_polygon'
						'keys_custom_fn': dict(shape_coef=lambda I, old_size, new_size: torch.stack((I[0]*(np.mean(np.array(new_size)/np.array(old_size))),I[1]),dim=0),
												image_px_in_nm=lambda x, old_size, new_size: x/(np.mean(np.array(new_size)/np.array(old_size)))),
						'size': (TEST_SIZE_HEIGHT, TEST_SIZE_WIDTH),
					}
				},
			],
			'MAX_NUM_CENTERS':2*1024,
		},
		'centerdir_gt_opts': dict(
			generic_regression_maps=[('shape_coef','gt_shape_coef')],

			ignore_instance_mask_and_use_closest_center=True,
			center_ignore_px=3,
			
			extend_instance_mask_weights=False,

			MAX_NUM_CENTERS=2*1024,
		),

		'batch_size': 1,
		'workers': 0,
	},

	model=dict(
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
	),
	center_model=dict(
		name='CenterAttributeEstimator',
		use_learnable_center_estimation=True,
		kwargs=dict(
			# define which attributes we regress: key is output name, values are input indexes from predicted maps (begining from the last center direction map)
			# key = shape_coef (as expected for CenterShapeEval)
			attributes=dict(shape_coef=dict(input_start=0, # start index for radius+circularity
								   			input_end=2, # end index for radius+circularity
                                   			use_log=True, 
											use_log_base='exp', # type of log used 
                                            use_log_channels=[0] # list of channels where to apply (relative to output channels for shape_coef)
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
		
	),
	num_vector_fields=NUM_FIELDS,
)

def get_args():
	return copy.deepcopy(args)
