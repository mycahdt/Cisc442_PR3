import math
import os
import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import as_strided


def sumOfAbsoluteDifferences(left_image, right_image):
	sad = np.sum(np.abs(left_image - right_image))
	return sad

def sumOfSquaredDifferences(left_image, right_image):
	ssd = np.sum(np.square(left_image.astype(np.float32) - right_image.astype(np.float32)))
	return ssd

def normalizedCrossCorrelation(left_image, right_image):
    meanA = np.mean(left_image)
    meanB = np.mean(right_image)	
    cross_corr = np.sum((left_image-meanA)*(right_image-meanB))

    # Compute the normalization factors
    norm_factor = np.sqrt(np.sum((left_image-meanA)**2) * np.sum((right_image-meanB)**2))
    
    if norm_factor == 0:
        return 0
	
	# Normalize the cross-correlation
    ncc = cross_corr / norm_factor
    return ncc




#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------


def sumOfSquaredDiff(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape
    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +window_size, strides=image.strides * 2)
    # Compute the sum of squared differences
    ssd = ((updatedImage - template) ** 2 * mask).sum(axis=-1).sum(axis=-1)
    return ssd


def disparity_ssd(left, right, template_x,template_y, window, lambdaValue):
    im_rows, im_cols = left.shape
    tpl_rows = template_y
    tpl_cols = template_x
    disparity = np.zeros(left.shape, dtype=np.float32)
    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype(np.float32)
            error = sumOfSquaredDiff(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaValue)
            _,_,min_loc,_ = cv.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def sumOfAbsDiff(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape
    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) + window_size, strides=image.strides * 2)
    # Compute the sum of squared differences
    ssd = ((abs(updatedImage - template)) * mask).sum(axis=-1).sum(axis=-1)
    return ssd


def disparity_sad(left, right, template_x,template_y, window, lambdaValue):
    im_rows, im_cols = left.shape
    tpl_rows = template_y
    tpl_cols = template_x
    disparity = np.zeros(left.shape, dtype=np.float32)
    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype(np.float32)
            error = sumOfAbsDiff(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaValue)
            _,_,min_loc,_ = cv.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def disparity_ncorr(left, right,template_x,template_y, window, lambdaValue):
    im_rows, im_cols = left.shape
    tpl_rows = template_y
    tpl_cols = template_x
    disparity = np.zeros(left.shape, dtype=np.float32)
    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype(np.float32)
            error = cv.matchTemplate(R_strip, tpl, method=cv.TM_CCORR_NORMED)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error - np.abs(dist * lambdaValue)
            _,_,_,max_loc = cv.minMaxLoc(cost)
            disparity[r, c] = dist[max_loc[0]]
    return disparity

def findCorners(image, window_size, k, thresh):
    # Find x and y derivatives
    dy, dx = np.gradient(image)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = image.shape[0]
    width = image.shape[1]
    cornerList = []
    copiedImage = image.copy()
    outputImage = cv.cvtColor(copiedImage, cv.COLOR_GRAY2RGB)
    offset = window_size//2
    # Loop through the images and detect the corners
    offset = int(offset)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            # Calculating corner response using determinant and trace
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            # If corner response crosses threshold, point is marked
            if r > thresh:
                # print x, y, r
                cornerList.append([x, y, r])
                outputImage.itemset((y, x, 0), 0)
                outputImage.itemset((y, x, 1), 0)
                outputImage.itemset((y, x, 2), 255)
    return outputImage, cornerList

def resolution(image, levels):
    h, w, c = image.shape
    outputImage = image
    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]
    return outputImage


def initImage(left,right,template_x,template_y, window):
    window_size = 3
    k = 0.15
    thresh = 100000
    #cornerList = cv2.cornerHarris(left, int(window_size), float(k), int(thresh))#2, 3, 0.04)
    finalLeft, cornerList = findCorners(left, int(window_size), float(k), int(thresh))
    finalRight, cornerList = findCorners(right, int(window_size), float(k), int(thresh))
    return finalLeft, finalRight, template_x,template_y, window


# Stereo matching using SSD
def ssd(in_left,in_right,template_x,template_y,window):
    # Calculate disparity maps of the left and right images
    left, right, template_x,template_y, window = initImage(in_left,in_right,template_x,template_y,window)
    left = cv.cvtColor(left,cv.COLOR_RGB2GRAY)
    right = cv.cvtColor(right, cv.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_ssd(left, right,template_x,template_y, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_ssd(right, left, template_x,template_y, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    rightDisparity = cv.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return leftDisparity, rightDisparity


# Stereo matching using SAD
def sad(in_left,in_right,template_x,template_y,window):
    # Calculate disparity maps of the left and right images
    left, right, template_x,template_y, window = initImage(in_left,in_right,template_x,template_y,window)
    left = cv.cvtColor(left, cv.COLOR_RGB2GRAY)
    right = cv.cvtColor(right, cv.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_sad(left, right, template_x,template_y, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_sad(right, left, template_x,template_y, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    rightDisparity = cv.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return leftDisparity, rightDisparity


# Stereo matching using normalized correlation
def ncc(in_left,in_right,template_x,template_y,window):
    # Calculate disparity maps of the left and right images
    left, right, template_x,template_y, window = initImage(in_left,in_right,template_x,template_y,window)
    left = cv.cvtColor(left, cv.COLOR_RGB2GRAY)
    right = cv.cvtColor(right, cv.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_ncorr(left, right, template_x,template_y, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_ncorr(right, left, template_x,template_y, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                        dtype=cv.CV_8U)
    rightDisparity = cv.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                        dtype=cv.CV_8U)
    return leftDisparity, rightDisparity



#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------



def average_neighborhood(disparity):

	print("Avg neighnorhood")

    # initialize a array of zeros of the size of the disparity map
	result = np.zeros_like(disparity, dtype=float)
    # Define the neighborhood size 
	neighborhood_size = 5
    # pad original array with zeros for 5x5 window and get shape 
	padDisp = np.pad(disparity, 2, constant_values=0)
	rows, cols = padDisp.shape
    # iterate through every pixel, except for the padding pixels
	for i in range(2, rows-2):
		for j in range(2, cols-2):
			if padDisp[i][j] == 0:
				neighborhood = padDisp[max(0, i - neighborhood_size // 2): min(rows, i + neighborhood_size // 2 + 1),
                                     max(0, j - neighborhood_size // 2): min(cols, j + neighborhood_size // 2 + 1)]
                # check how many non-zeros are in the window
				nrows, ncols = neighborhood.shape
				ints = 0
				for r in range(nrows):
					for c in range(ncols):
						if neighborhood[r][c] > 0:
							ints += 1
				if ints >= ((neighborhood_size*neighborhood_size)/neighborhood_size):
					avg = (np.sum(neighborhood)/ints)
					result[i-2][j-2] = avg
				else:
					continue
			else:
				result[i-2][j-2] = padDisp[i][j] # -2 in the result bc it isn't padded
	return result
	


def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, disparity):
	
	# Initialize disparity map
	height, width = left_image.shape
	disparity = np.zeros_like(left_image, dtype=np.float32)

    # Define half template sizes
	half_template_size_x = TEMPLATE_SIZE_X // 2
	half_template_size_y = TEMPLATE_SIZE_Y // 2
	
	for y in range(half_template_size_y, height - half_template_size_y):
		for x in range(half_template_size_x, width - half_template_size_x):
			template_left = left_image[y - half_template_size_y:y + half_template_size_y + 1, x - half_template_size_x:x + half_template_size_x + 1]
			
			if DISTANCE == "NCC":
				min_cost = -1
			else:
				min_cost = float('inf')

			best_disparity = 0
			
			for d in range(max(0, x - SEARCH_RANGE), min(width, x + SEARCH_RANGE + 1)):
				template_right = right_image[y - half_template_size_y:y + half_template_size_y + 1,d - half_template_size_x:d + half_template_size_x + 1]


				if template_left.shape != template_right.shape:
					continue

                # Compute sum of absolute differences (SAD) as cost
				cost = 0

				if DISTANCE == "SAD":
					cost = sumOfAbsoluteDifferences(template_left, template_right)
				elif DISTANCE == "SSD":
					cost = sumOfSquaredDifferences(template_left, template_right)
				else:
					cost = normalizedCrossCorrelation(template_left, template_right)
				
				if DISTANCE == "SAD" or DISTANCE == "SSD":
					if cost < min_cost:
						min_cost = cost
						best_disparity = abs(x - d)
				else:
					if cost > min_cost:
						min_cost = cost
						best_disparity = abs(x - d)
				

            # Assign the best disparity to the disparity map
			disparity[y, x] = best_disparity

	return disparity

def feature_based(right_image, left_image, distance, searchRange, templateSizeX, templateSizeY, disparity):
	if distance == "SAD":
		left, right = sad(left_image, right_image, templateSizeX, templateSizeY, searchRange)
	elif distance == "SSD":
		left, right = ssd(left_image, right_image, templateSizeX, templateSizeY, searchRange)
	else:
		left, right = ncc(left_image, right_image, templateSizeX, templateSizeY, searchRange)
	left = validityFeature(left, right)
	return left






# Validity check of the two images
def validityFeature(left, right):
    r1, c1 = left.shape
    r2, c2 = right.shape
    # Validate left image by calculating left - right image disparities
    for i in range(0, r1, 1):
        for j in range(0, c1, 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0
    # Validate left image by calculating right - left image disparities
    for i in range(0, r2, 1):
        for j in range(0, c2, 1):
            if right[i, j] != left[i, j]:
                right[i, j] = 0
    return left


def validityCheck(method, left_image, right_image, distance, searchRange, templateSizeX, templateSizeY, disparity):
	
	if method == 'region':
		rightDisparity = region_based(right_image, left_image, distance, searchRange, templateSizeX, templateSizeY, 1)
		print("Final disparity:", disparity)
	else:
		rightDisparity = feature_based(right_image, left_image, distance, searchRange, templateSizeX, templateSizeY, 1)
    
	matches = []
     
	for i in range(disparity.shape[0]):
		for j in range(disparity.shape[1]):
			if rightDisparity[i, j] == disparity[i, j]:
				matches.append((i, j))
			else:
				disparity[i, j] = 0
	return disparity  


def disparityMap(left_image_path, right_image_path):
     

	left = cv.imread(left_image_path)
	right = cv.imread(right_image_path)

	left_image=cv.cvtColor(left, cv.COLOR_BGR2GRAY)
	right_image=cv.cvtColor(right, cv.COLOR_BGR2GRAY)
     
	distance = input('Enter distance [SAD, SSD, NCC]: ')
	method = input('Enter method [region, feature]: ')
	searchRange = int(input('Enter search range (need to be integer): '))
	templateSizeX = int(input('Enter template_x_size (need to be odd integer): '))
	templateSizeY = int(input('Enter template_y_size (need to be odd integer): '))


	if method == 'region':
		disparity = region_based(left_image, right_image, distance, searchRange, templateSizeX, templateSizeY, 1)
		print("Final disparity:", disparity)

		disparity = validityCheck(method, left_image, right_image, distance, searchRange, templateSizeX, templateSizeY, disparity)
     
		for i in range(2):
			disparity = average_neighborhood(disparity)

		disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
	elif method == 'feature':
		disparity = feature_based(left_image, right_image, distance, searchRange, templateSizeX, templateSizeY, 1)
		
    
	return disparity
     


def main():
     #-------------------------------------------------------------------------------------------------------------------
     # Venus Images --> Venus 2 and Venus 6

     left_image_path = "/Users/mycahdetorres/Downloads/Project3/images/venus2.jpeg"
     right_image_path = "/Users/mycahdetorres/Downloads/Project3/images/venus6.jpeg"

	 # region_disparityVenus_SSD.png
     # Distance = SSD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3 
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityVenus_SSD.png', disparity)
     '''
     
     # region_disparityVenus_SAD.png
     # Distance = SAD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityVenus_SAD.png', disparity)
     ''' 

	 # region_disparityVenus_NCC.png
     # Distance = NCC, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3 
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityVenus_NCC.png', disparity)
     '''
     



	 #-------------------------------------------------------------------------------------------------------------------
     # Bull Images --> Bull 2 and Bull 6
     left_image_path = "/Users/mycahdetorres/Downloads/Project3/images/bull2.jpeg"
     right_image_path = "/Users/mycahdetorres/Downloads/Project3/images/bull6.jpeg"

	 # region_disparityBull_SSD.png
     # Distance = SSD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3 
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityBull_SSD.png', disparity)
     '''
     

     # region_disparityBull_SAD.png
     # Distance = SAD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityBull_SAD.png', disparity)
     '''
     
	 # region_disparityBull_NCC.png
     # Distance = NCC, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityBull_NCC.png', disparity)
     '''


	 #-------------------------------------------------------------------------------------------------------------------
     # Poster Images --> Poster 2 and Poster 6
     left_image_path = "/Users/mycahdetorres/Downloads/Project3/images/poster2.jpeg"
     right_image_path = "/Users/mycahdetorres/Downloads/Project3/images/poster6.jpeg"

	 # region_disparityPoster_SSD.png
     # Distance = SSD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3 
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityPoster_SSD.png', disparity)
     '''
     

     # region_disparityPoster_SAD.png
     # Distance = SAD, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityPoster_SAD.png', disparity)
     '''
     
	 # region_disparityPoster_NCC.png
     # Distance = NCC, Method = region, search Range = 20, templateSizeX = 3, templateSizeY = 3
     '''
     disparity = disparityMap(left_image_path, right_image_path)
     cv.imwrite('output/region_disparityPoster_NCC.png', disparity)
     '''
     

	 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 # Feature Based

	 #-------------------------------------------------------------------------------------------------------------------
     # Barn1 Images --> Barn1_2 and Barn1_6
     left_image_path = "/Users/mycahdetorres/Downloads/Project3/images/barn1_2.jpeg"
     right_image_path = "/Users/mycahdetorres/Downloads/Project3/images/barn1_6.jpeg" 
	 
	 # feature_disparityBarn1_SSD.png
	 # Distance = SSD, Method = feature, search Range = 90, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn1_SSD.png', disparity)
	 

     # feature_disparityBarn1_SAD.png
     # Distance = SAD, Method = feature, search Range = 90, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn1_SAD.png', disparity)


	 # feature_disparityBarn1_NCC.png
     # Distance = NCC, Method = feature, search Range = 90, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn1_NCC.png', disparity)
    
	 #-------------------------------------------------------------------------------------------------------------------
     # Barn2 Images --> Barn2_2 and Barn2_6
     left_image_path = "/Users/mycahdetorres/Downloads/Project3/images/barn2_2.jpeg"
     right_image_path = "/Users/mycahdetorres/Downloads/Project3/images/barn2_6.jpeg"

	 # feature_disparityBarn2_SSD.png
     # Distance = SSD, Method = feature, search Range = 100, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn2_SSD.png', disparity)

     # feature_disparityBarn2_SAD.png
     # Distance = SAD, Method = feature, search Range = 100, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn2_SAD.png', disparity)

     
	 # feature_disparityBarn2_NCC.png
     # Distance = NCC, Method = feature, search Range = 100, templateSizeX = 11, templateSizeY = 11
     #disparity = disparityMap(left_image_path, right_image_path)
     #cv.imwrite('output/feature_disparityBarn2_NCC.png', disparity)

	
     

     

    
    

 
if __name__ == '__main__':
    main()
