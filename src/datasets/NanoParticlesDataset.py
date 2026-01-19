import glob
import os, cv2

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import sys, os

import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import transforms as my_transforms

class NanoParticlesDataset(Dataset):
	DATASET_NAME = 'nanoparticles'

	IGNORE_LABEL = "ignore-region"

	IGNORE_FLAG = 1
	IGNORE_TRUNCATED_FLAG = 2
	IGNORE_OVERLAP_BORDER_FLAG = 4
	IGNORE_DIFFICULT_FLAG = 8
	
	def __init__(self, name_pattern, gt_name_replace_args, root_dir='./', subfolders=None, gt_optional=False,
			   gt_from_circles_fitting=True, gt_from_elipse_fitting=False, gt_from_polygons=False,
			   gt_from_circularity=False, gt_circularity_inverse=False, mapping_to_nm=None, 
			   gt_border_centers_and_label_fix=False,
			   BORDER_MARGIN_FOR_MASK=0, keep_centers_at_border_margin=False, mark_truncated_center_box_for_border_margin=False, mark_truncated_mask_for_border_margin=True,
			   fixed_bbox_size=15, resize_factor=None, MAX_NUM_CENTERS=1024, transform=None, valid_sample_names=None, 
			   num_cpu_threads=1, check_consistency=True, name_pattern_for_channels=None,
				transform_only_valid_centers=False, transform_per_sample_rng=False, **kwargs):
		print('NanoParticlesDataset created')

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.gt_optional = gt_optional

		self.gt_from_circles_fitting = gt_from_circles_fitting
		self.gt_from_elipse_fitting = gt_from_elipse_fitting
		self.gt_from_circularity = gt_from_circularity
		self.gt_from_polygons = gt_from_polygons

		self.gt_circularity_inverse = gt_circularity_inverse

		self.gt_border_centers_and_label_fix = gt_border_centers_and_label_fix

		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor

		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

		self.transform = my_transforms.get_transform(transform) if type(transform) == list else transform
		self.rng = np.random.default_rng(1337)
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng

		self.return_image = True
		self.check_consistency = check_consistency

		self.mapping_to_nm = mapping_to_nm

		self.gt_name_replace_args = gt_name_replace_args
		self.name_pattern_for_channels = name_pattern_for_channels

		sample_list = []

		if subfolders is None:
			subfolders = ["*/*"]
		if type(subfolders) not in [list, tuple]:
			subfolders = [subfolders]

		if type(name_pattern) not in [list,tuple]:
			name_pattern = [name_pattern]

		for pattr in name_pattern:
			for sub in subfolders:
				sample_list += sorted(glob.glob(os.path.join(root_dir,sub,pattr)))

		if valid_sample_names is not None:
			def filter_by_name(x):
				return any([v in x for v in valid_sample_names])
			
			sample_list = list(filter(filter_by_name, sample_list))
		
		self.sample_list = sample_list

		self.size = len(self.sample_list)
		print(f'NanoParticlesDataset of size {len(sample_list)}')        

		self.remove_out_of_bounds_centers = True
		self.BORDER_MARGIN_FOR_CENTER = 0
		self.BORDER_MARGIN_FOR_MASK = BORDER_MARGIN_FOR_MASK
		self.keep_centers_at_border_margin = keep_centers_at_border_margin
		self.mark_truncated_center_box_for_border_margin = mark_truncated_center_box_for_border_margin
		self.mark_truncated_mask_for_border_margin = mark_truncated_mask_for_border_margin

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		im_fn = self.sample_list[index]

		if self.name_pattern_for_channels is None:
			image = Image.open(im_fn).convert('RGB')
		else:
			# default image (we assume grayscale)
			image = Image.open(im_fn).convert('L')
			
			channel_imgs = [np.array(image)]			

			assert len(self.name_pattern_for_channels) < 3

			# load all other per-channel images
			for args in self.name_pattern_for_channels:
				im_fn_channel = im_fn.replace(*args)

				image_channel = Image.open(im_fn_channel).convert('L')
				channel_imgs.append(np.array(image_channel))
			
			while len(channel_imgs) < 3:
				channel_imgs.append(np.zeros_like(image))
		
			image = np.stack(channel_imgs,axis=2)

			image = Image.fromarray(image.astype(np.uint8))



		im_size = image.size
		org_im_size = np.array(image.size)

		if self.resize_factor is not None:
			im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

		# avoid loading full buffer data if image not requested
		if self.return_image:
			if self.resize_factor is not None and self.resize_factor != 1.0:
				image = image.resize(im_size, Image.BILINEAR)
		else:
			image = None

		sample = dict(image=image,
					  im_name=im_fn,
					  org_im_size=org_im_size,
					  im_size=im_size,
					  index=index)

		num_shape_coef = 0
		# CAUTION: order of IFs must match order in for loop !!
		if self.gt_from_elipse_fitting:
			num_shape_coef = 3
		if self.gt_from_circles_fitting:
			num_shape_coef = 1
		if self.gt_from_circularity:
			num_shape_coef = 2

		label = torch.zeros((im_size[1], im_size[0]), dtype=torch.uint8)
		instances = torch.zeros((im_size[1], im_size[0]), dtype=torch.int16)
		shape_coef = torch.zeros(num_shape_coef, im_size[1], im_size[0], dtype=torch.float32)
		ignore = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)
		truncated = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)
		
		# read centers from numpy file
		gt_fn = im_fn
		for args in self.gt_name_replace_args:
			gt_fn = gt_fn.replace(*args)

		if os.path.exists(gt_fn) or self.gt_optional == False:
			if gt_fn.endswith(".npy"):
				masks = np.load(gt_fn)
			elif gt_fn.endswith(".npz"):
				masks = np.load(gt_fn)['arr_0']
			else:
				raise Exception("Unsupported input data, expected .npy or .npz files")
		else:
			masks =  torch.zeros((im_size[1], im_size[0],0), dtype=torch.uint8)
		

		if "_masks." in gt_fn:
			padding_fn = gt_fn.replace("_masks.","_padding.")
			if os.path.exists(padding_fn):
				# load padding mask
				if padding_fn.endswith(".npy"):
					padding_mask = np.load(padding_fn)
				elif padding_fn.endswith(".npz"):
					padding_mask = np.load(padding_fn)['arr_0']
				else:
					raise Exception("Unsupported input data, expected .npy or .npz files")
				
				if self.resize_factor is not None and self.resize_factor != 1.0:
					padding_mask = np.array(Image.fromarray(image).resize(padding_mask, Image.BILINEAR))

				truncated[0][padding_mask == 0] = self.IGNORE_FLAG

		if self.BORDER_MARGIN_FOR_MASK > 0:
			# increase ignore region for self.BORDER_MARGIN_FOR_MASK pixels
			ignore_mask = truncated[0] == self.IGNORE_FLAG

			# Change the border values to 255
			ignore_mask[0, :] = 1  # Set the top row to 255
			ignore_mask[-1, :] = 1  # Set the bottom row to 255
			ignore_mask[:, 0] = 1  # Set the left column to 255
			ignore_mask[:, -1] = 1  # Set the right column to 255

			kernel = np.ones((3, 3), np.uint8)
			ignore_mask = cv2.dilate(ignore_mask.cpu().numpy().astype(np.uint8), kernel, iterations=self.BORDER_MARGIN_FOR_MASK)

			truncated[0][ignore_mask == 1] = self.IGNORE_FLAG
			ignore_ids = np.where(truncated[0].flatten())[0]

		M = self.fixed_bbox_size
		instance_counter = 1
		
		centers = []
		gt_polygon_list = []
		for i in range(masks.shape[2]):

			if not self.gt_border_centers_and_label_fix:
				# mark as truncated any particle that intersect with the dilated ignore region
				if self.BORDER_MARGIN_FOR_MASK > 0:
					from utils.overlaps import overlap_pixels_ids
					mask_ids = np.where(masks[:,:,i].flatten()!=0)[0]
					iou = overlap_pixels_ids(mask_ids,ignore_ids)

					if iou > 0:
						# mark as truncated
						ignore[0][masks[:,:,i]!=0] = self.IGNORE_TRUNCATED_FLAG
						# do not add it to final list of centerss
						continue
			
			if self.gt_from_circularity or self.gt_from_polygons:
				contours, _ = cv2.findContours((masks[:,:,i]!=0).astype(np.uint8), 
											cv2.RETR_EXTERNAL,  # Only external contours
											cv2.CHAIN_APPROX_SIMPLE)

			# check if pt is outside of bounds and ignore this one if needed
			if self.gt_from_polygons:
				cnt = [np.array(c).reshape(-1, 2) for c in contours if len(c) > 4]
				if len(cnt ) <= 0:
					print(f"WARNING: Found zero polygons for a single annotated object - skipping {i}-th object in '{im_fn}'")
					continue

				cnt = np.array(self.merge_multiple_contours(cnt)).reshape(-1,2)


				if self.resize_factor is not None and self.resize_factor != 1.0:
					cnt = cnt * self.resize_factor

				gt_polygon_list.append(cnt)

			if self.gt_from_elipse_fitting:
				ellipsis_fit = self.fit_ellipses(masks[:,:,i].astype(np.uint8)*255, limit_size=200)
				
				if len(ellipsis_fit) > 1:
					print(ellipsis_fit)
					print(f"WARNING: Found multiple possible elipses for a single annotated object (expected only one solution!) in '{im_fn}'")
				elif len(ellipsis_fit) < 1:
					print(ellipsis_fit)
					print(f"WARNING: Found zero elipses for a single annotated object - skipping {i}-th object in '{im_fn}'")
					continue

				ellipsis_fit = ellipsis_fit[0] # should be only one circle
				
				pt = [ellipsis_fit[0],ellipsis_fit[1]]
				coef = np.array([float(ellipsis_fit[2]),
					 				float(ellipsis_fit[3]),
									float(ellipsis_fit[4])])

				# clip locations within image bounds
				#pt[0] = np.clip(pt[0], 0, masks.shape[-2]-1)
				#pt[1] = np.clip(pt[1], 0, masks.shape[-3]-1)
			if self.gt_from_circles_fitting:
				circles_fit = self.fit_circles(masks[:,:,i].astype(np.uint8)*255)
				
				if len(circles_fit) > 1:
					print(circles_fit)
					print(f"WARNING: Found multiple possible circles for a single annotated object (expected only one solution!) in '{im_fn}'")
				elif len(circles_fit) < 1:
					print(circles_fit)
					print(f"WARNING: Found zero circles for a single annotated object - skipping {i}-th object in '{im_fn}'")
					continue

				circles_fit = circles_fit[0] # should be only one circle

				pt = [circles_fit[0],circles_fit[1]]
				coef = np.array([float(circles_fit[-1])])

				# clip locations within image bounds
				#pt[0] = np.clip(pt[0], 0, masks.shape[-2]-1)
				#pt[1] = np.clip(pt[1], 0, masks.shape[-3]-1)

			if self.gt_from_circularity:
				perimiter = sum([len(c) for c in contours if len(c) > 4])				
				area = masks[:,:,i].sum()

				circularity = 4*np.pi* (area/perimiter**2)
				if self.gt_circularity_inverse:
					circularity = 1-1/circularity
				
				
				# set radii from the size of mask with the assumption of having a circle
				coef = np.array([float(np.sqrt(area / np.pi)), # radii from area of circle
					 			 float(circularity)])
			
			if not self.gt_from_elipse_fitting and not self.gt_from_circles_fitting:
				# find non-zero values a
				non_zero_loc = np.where(masks[:,:,i])
				pt = np.mean(non_zero_loc,axis=1)
				pt = [pt[1],pt[0]]
		
			if self.resize_factor is not None and self.resize_factor != 1.0:
				pt = pt * self.resize_factor

			mask_pt = [(np.clip(int(pt[1] - M),0, label.shape[0]), np.clip(int(pt[1] + M),0, label.shape[0])),
			  			(np.clip(int(pt[0] - M),0, label.shape[0]), np.clip(int(pt[0] + M),0, label.shape[0]))]
			
			if self.gt_border_centers_and_label_fix:
				# mark as truncated any particle that intersect with the dilated ignore region
				if self.BORDER_MARGIN_FOR_MASK > 0:
					from utils.overlaps import overlap_pixels_ids
					mask_ids = np.where(masks[:,:,i].flatten()!=0)[0]
					iou = overlap_pixels_ids(mask_ids,ignore_ids)

					if iou > 0:						
						if self.mark_truncated_mask_for_border_margin:
							# mark whole mask as truncated
							ignore[0][masks[:,:,i]!=0] = self.IGNORE_TRUNCATED_FLAG
						if self.mark_truncated_center_box_for_border_margin:
							# also mark around center to ignore
							ignore[0][mask_pt[0][0]:mask_pt[0][1], mask_pt[1][0]:mask_pt[1][1]] = self.IGNORE_TRUNCATED_FLAG
						if not self.keep_centers_at_border_margin:
							# do not add it to final list of centers
							continue

			  
			label[mask_pt[0][0]:mask_pt[0][1], mask_pt[1][0]:mask_pt[1][1]] = 1
			instances[mask_pt[0][0]:mask_pt[0][1], mask_pt[1][0]:mask_pt[1][1]] = instance_counter
			shape_coef[:,mask_pt[0][0]:mask_pt[0][1], mask_pt[1][0]:mask_pt[1][1]] = torch.from_numpy(coef).reshape(-1,1,1)
			if not self.gt_border_centers_and_label_fix:
				label[masks[:,:,i] != 0] = 1

			instance_counter += 1

			centers.append(pt)

		centers = np.array(centers)

		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		if len(centers) > 0:
			sample['center'][1:centers.shape[0]+1, :] = centers[:,:2]
		sample['label'] = label.unsqueeze(0)
		sample['mask'] = (label > 0).unsqueeze(0)
		sample['ignore'] = ignore
		sample['instance'] = instances.unsqueeze(0)
		sample['shape_coef'] = shape_coef
		sample['name'] = im_fn
		if len(gt_polygon_list) > 0:
			from utils.utils import instance_poly_to_variable_array
			sample['instance_polygon'] = instance_poly_to_variable_array(gt_polygon_list)

		if self.mapping_to_nm is not None:
			sample['image_px_in_nm'] = self.mapping_to_nm(im_fn) 
			
			if self.resize_factor is not None and self.resize_factor != 1.0:
				sample['image_px_in_nm'] *= self.resize_factor

		if self.transform is not None:
			import copy
			do_transform = True

			ii = 0
			while do_transform:			
				if ii > 10 and ii % 10 == 0:
					print(f"WARNING: unable to generate valid transform for {ii} iterations")
				new_sample = self.transform(copy.deepcopy(sample), self.rng if not self.transform_per_sample_rng else np.random.default_rng(1337))

				out_of_bounds_ids = [id for id, c in enumerate(new_sample['center']) if c[0] < 0 or c[1] < 0 or c[0] >= new_sample['image'].shape[-1] or c[1] >= new_sample['image'].shape[-2]]

				# stop if sufficent centers still visible
				if not self.transform_only_valid_centers or self.transform_only_valid_centers <= 0:
					do_transform = False
					sample = new_sample
				else:
					if type(self.transform_only_valid_centers) == bool:
						if len(out_of_bounds_ids) < len(centers): # at least one must be present
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == int:
						if len(centers) - len(out_of_bounds_ids) >= self.transform_only_valid_centers:
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == float:
						min_visible = int(self.transform_only_valid_centers * len(centers))
						if len(centers) - len(out_of_bounds_ids) >= min_visible:
							do_transform = False
							sample = new_sample
					else:
						raise Exception("Invalid type of transform_only_valid_centers, allowed types: bool, int, float")				
				ii += 1

		if self.remove_out_of_bounds_centers:
			# if instance has out-of-bounds center then ignore it if requested so
			out_of_bounds_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id > 0 and (c[0] < 0 or c[1] < 0 or
											c[0] >= sample['image'].shape[-1] or
											c[1] >= sample['image'].shape[-2])]
		
			for id in out_of_bounds_ids:
				sample['instance'][sample['instance'] == id] = 0
				sample['center'][id,:] = -1
		
		
		if self.transform is not None and 'instance' in sample:
			# recheck for ignore regions due to changes from augmentation:
			#  - mark with ignore any centers/instances that are now outside of image size
			valid_ids = np.unique(sample['instance'])
			truncated_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id > 0 and id in valid_ids and (c[0] < self.BORDER_MARGIN_FOR_CENTER or
																c[1] < self.BORDER_MARGIN_FOR_CENTER or
																c[0] >= sample['image'].shape[-1]-self.BORDER_MARGIN_FOR_CENTER or
																c[1] >= sample['image'].shape[-2]-self.BORDER_MARGIN_FOR_CENTER)]
			for id in truncated_ids:
				sample['ignore'][sample['instance'] == id] |= self.IGNORE_TRUNCATED_FLAG

		if self.check_consistency:
			if np.any(np.unique(instances) != torch.unique(sample['instance']).numpy()):
				print(f"error in augmentation - missing one or more sample in instance mask after augmentation (before was {np.unique(instances)} now is {torch.unique(sample['instance'])})")

		return sample

	def fit_circles(self, mask, limit_size=200):
		from scipy.optimize import least_squares

		def circle_residuals(params, points):
			# params contains [xc, yc, r], where (xc, yc) is the center, r is the radius
			xc, yc, r = params
			return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r

		# Step 1: Find contours
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		circles = []
		# Step 2: Loop through contours and fit a circle to each one
		for contour in contours:
			if len(contour) < 4:
				continue
			# Get the points from the contour
			contour_points = np.array(contour).reshape(-1, 2)
			
			# Initial guess for the center (mean of contour points) and radius (average distance from center)
			x0 = np.mean(contour_points[:, 0])
			y0 = np.mean(contour_points[:, 1])
			r0 = np.mean(np.sqrt((contour_points[:, 0] - x0)**2 + (contour_points[:, 1] - y0)**2))

			lower_bounds = [-np.inf, -np.inf, 0]  # Lower bounds for [xc, yc, r]
			upper_bounds = [+np.inf, +np.inf, limit_size]  # Upper bounds for [xc, yc, r]

			# Perform least squares fitting to find the best circle
			result = least_squares(circle_residuals, [x0, y0, r0], args=(contour_points,),bounds=(lower_bounds, upper_bounds))

			# Extract the fitted circle parameters
			xc, yc, r = result.x

			# ensure radius found has at least 2 pixles
			if r > 2:
				circles.append(result.x)
		
		return circles
	
	def fit_ellipses(self, mask, limit_size=np.inf, fit_circle_as_init=False):
		from scipy.optimize import least_squares

		def ellipse_residuals(params, points):
			# params contains [xc, yc, a, b, theta], where (xc, yc) is the center, 
			# a is the major axis length, b is the minor axis length, and theta is the rotation angle
			xc, yc, a, b, theta = params
			
			# Rotate points by the angle theta
			x_rot = (points[:, 0] - xc) * np.cos(theta) + (points[:, 1] - yc) * np.sin(theta)
			y_rot = -(points[:, 0] - xc) * np.sin(theta) + (points[:, 1] - yc) * np.cos(theta)
			
			# Ellipse equation: (x' / a)^2 + (y' / b)^2 = 1
			residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
			return residuals

		def circle_residuals(params, points):
			# params contains [xc, yc, r], where (xc, yc) is the center, r is the radius
			xc, yc, r = params
			return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r


		# Step 1: Find contours
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		ellipses = []
		# Step 2: Loop through contours and fit an ellipse to each one
		for contour in contours:
			if len(contour) < 4:
				continue
			# Get the points from the contour
			contour_points = np.array(contour).reshape(-1, 2)
			
			# fit circle as initial guess
			if fit_circle_as_init:
				# Initial guess for the center (mean of contour points) and radius (average distance from center)
				x0 = np.mean(contour_points[:, 0])
				y0 = np.mean(contour_points[:, 1])
				r0 = np.mean(np.sqrt((contour_points[:, 0] - x0)**2 + (contour_points[:, 1] - y0)**2))

				# Perform least squares fitting to find the best circle
				result = least_squares(circle_residuals, [x0, y0, r0], args=(contour_points,))

				xc,yc,a0 = result.x
				b0 = a0
				theta0 = 0
			else:
				# Initial guess for the center (mean of contour points), axes lengths (based on bounding box), and rotation (0)
				xc = np.mean(contour_points[:, 0])
				yc = np.mean(contour_points[:, 1])

				# Compute the bounding box of the contour
				min_x, min_y = np.min(contour_points, axis=0)
				max_x, max_y = np.max(contour_points, axis=0)
				
				# Use half of the width and height of the bounding box as the initial guess for axes lengths
				a0 = np.mean(np.sqrt((contour_points[:, 0] - xc)**2 + (contour_points[:, 1] - yc)**2)) #(max_x - min_x) / 2.0
				b0 = np.mean(np.sqrt((contour_points[:, 0] - xc)**2 + (contour_points[:, 1] - yc)**2)) #(max_y - min_y) / 2.0

				# Initial guess for rotation (set to 0 for simplicity, this can be improved)
				theta0 = 0.0

			lower_bounds = [-np.inf, -np.inf, 0, 0, -np.pi]  # Lower bounds for [xc, yc, a, b, theta]
			upper_bounds = [+np.inf, +np.inf, limit_size, limit_size, np.pi]  # Upper bounds for [xc, yc, a, b, theta]
		

			# Perform least squares fitting to find the best ellipse
			result = least_squares(ellipse_residuals, [xc, yc, a0, b0, theta0], args=(contour_points,), bounds=(lower_bounds, upper_bounds))

			# Extract the fitted ellipse parameters
			xc, yc, a, b, theta = result.x

			# Ensure that the axes have valid sizes (both axes must be at least 2 pixels)
			if a > 2 and b > 2:
				ellipses.append(result.x)
		
		return ellipses

	@staticmethod
	def merge_multiple_contours(cnt):
		def interpolate_line(start,end):
			S, E = np.array(start), np.array(end)

			# iterate over longest axis
			dim = np.argmax(np.abs(S - E))
			if dim > 0:
				# switch axis
				S, E = S[::-1], E[::-1]

			k = (E[1] - S[1]) / float(E[0] - S[0])
			n = S[1] - k * S[0]

			x = np.arange(S[0],E[0],step=1 if E[0] > S[0] else -1)
			y = np.round(k*x+n).astype(np.int32)
			points = np.stack((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
			points = points[:,::-1 if dim > 0 else 1]
			points = points[1:] if len(points) > 0 else points
			return np.array(points).reshape(-1,2)

		cnt = cnt.copy()
		# sort list by num of points
		idx = np.argsort([len(C) for C in cnt])
		# select longest contour and remove it from the list (i.e. first one when sorted desc)
		c = cnt.pop(idx[-1])

		# remove ones with only 2 points
		cnt = [C for C in cnt if len(C) > 2]
		while len(cnt) > 0:
			# calc distances to all points between current and remaining contours
			contour_dists = [np.linalg.norm(c[:, None, :] - c_i[None, :, :], axis=-1) for c_i in cnt]

			# select contour with smallest distance
			i = np.argmin([d.min() for d in contour_dists])
			dist = contour_dists[i]
			c_i = cnt.pop(i)

			# find closest two points between contours
			min_c, min_i = np.unravel_index(np.argmin(dist),dist.shape)

			# splice contours where points are closest
			c_i = np.roll(c_i,-min_i,axis=0)

			# ensure at least one element is left after inserted splice
			if min_c == len(c) -1:
				c = np.roll(c, -1, axis=0)
				min_c -= 1

			start_splice = interpolate_line(c[min_c],c_i[0])
			end_splice = interpolate_line(c_i[-1], c[min_c+1])

			c = np.concatenate((c[:min_c+1],start_splice,c_i,
								c_i[:1],end_splice,c[min_c:min_c+1],c[min_c:]),axis=0)

		return c
