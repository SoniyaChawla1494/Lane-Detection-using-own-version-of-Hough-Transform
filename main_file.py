
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

prewX = (1.0 / 3.0) * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

prewY = (1.0 / 3.0) * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

def horizontalPrewitt(img, row, column):
    sum = 0
    hor = 0
    for i in range(0, 3):
        for j in range(0, 3):
            hor += img[row + i, column + j] * prewX[i, j]
    return hor

def verticalPrewitt(img, row, column):
    sum = 0
    ver = 0
    for i in range(0, 3):
        for j in range(0, 3):
            ver += img[row + i, column + j] * prewY[i, j]
    return ver

def gradient_in_X(img, height, width):
    result_img = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, img[i].size - 5):
            if ifContainedInRegion(img, i, j):
                result_img[i + 1][j + 1] = None
            else:
                result_img[i + 1][j + 1] = horizontalPrewitt(img, i, j)

    return abs(result_img)

def gradient_in_Y(imgArr, height, width):
    result_img = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if ifContainedInRegion(imgArr, i, j):
                result_img[i + 1][j + 1] = None
            else:
                result_img[i + 1][j + 1] = verticalPrewitt(imgArr, i, j)

    return abs(result_img)

def ifContainedInRegion(image, i, j):
    return image[i][j] == None or image[i][j + 1] == None or image[i][j - 1] == None or image[i + 1][j] == None or \
           image[i + 1][j + 1] == None or image[i + 1][j - 1] == None or image[i - 1][j] == None or \
           image[i - 1][j + 1] == None or image[i - 1][j - 1] == None

def calImageMagnitude(Gradient_x, Gradient_Y, height, width):
    grad_values = np.empty(shape=(height, width))
    for row in range(height):
        for column in range(width):
            grad_values[row][column] = ((Gradient_x[row][column] ** 2 + Gradient_Y[row][column] ** 2) ** 0.5) / 1.4142
    return grad_values

def get_gradient_angle(Gradient_X, Gradient_Y, height, width):
    gradientData = np.empty(shape=(height, width))
    angle = 0
    for i in range(height):
        for j in range(width):
            if Gradient_X[i][j] == 0:
                if Gradient_Y[i][j] > 0:
                    angle = 90
                else:
                    angle = -90
            else:
                angle = math.degrees(math.atan(Gradient_Y[i][j] / Gradient_X[i][j]))
            if angle < 0:
                angle += 360
            gradientData[i][j] = angle
    return gradientData


def findLocalMaxima(gradient_data, gradient_angle, height, width):
    gradient = np.empty(shape=(height, width))
    num_pixs = np.zeros(shape=(256))
    pixelsAtEdge = 0

    for row in range(5, height - 5):
        for col in range(5, gradient_data[row].size - 5):
            angle = gradient_angle[row, col]
            gradientAtPixel = gradient_data[row, col]
            value = 0

            if (0 <= angle <= 22.5 or 157.5 < angle <= 202.5 or 337.5 < angle <= 360):
                if gradientAtPixel > gradient_data[row, col + 1] and gradientAtPixel > gradient_data[row, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            elif (22.5 < angle <= 67.5 or 202.5 < angle <= 247.5):
                if gradientAtPixel > gradient_data[row + 1, col - 1] and gradientAtPixel > gradient_data[
                    row - 1, col + 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            elif (67.5 < angle <= 112.5 or 247.5 < angle <= 292.5):
                if gradientAtPixel > gradient_data[row + 1, col] and gradientAtPixel > gradient_data[row - 1, col]:
                    value = gradientAtPixel
                else:
                    value = 0

            elif 112.5 < angle <= 157.5 or 292.5 < angle <= 337.5:
                if gradientAtPixel > gradient_data[row + 1, col + 1] \
                        and gradientAtPixel > gradient_data[row - 1, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            gradient[row, col] = value

            if value > 0:
                pixelsAtEdge += 1
                try:
                    num_pixs[int(value)] += 1
                except:
                    print('Values out of range', value)

    print('Count of Edge pixels:', pixelsAtEdge)
    return [gradient, num_pixs, pixelsAtEdge]


def threshold_image(percentage, image_data, noPixs, edge_pixs, file):
    thresh = np.around(edge_pixs * percentage / 100)
    sum, value = 0, 255
    for value in range(255, 0, -1):
        sum += noPixs[value]
        if sum >= thresh:
            break

    for i in range(image_data.shape[0]):
        for j in range(image_data[i].size):
            if image_data[i, j] < value:
                image_data[i, j] = 0
            else:
                image_data[i, j] = 255

    print('Done untill', percentage, '- result:')
    print('Total number of pixels after thresholding:', sum)
    print('Gray level values after thresholding:', value)
    return image_data


def canny_detection(image):
    height = image.shape[0]
    width = image.shape[1]

    Gradient_x = gradient_in_X(image, height, width)
    Gradient_y = gradient_in_Y(image, height, width)
    
    gradient_value = calImageMagnitude(Gradient_x, Gradient_y, height, width)

    gradient_angle = get_gradient_angle(Gradient_x, Gradient_y, height, width)

    suppressedLocalMax = findLocalMaxima(gradient_value, gradient_angle, height, width)

    suppressed_image = suppressedLocalMax[0]
    num_pixs = suppressedLocalMax[1]
    edge_pixs = suppressedLocalMax[2]

    result_image = threshold_image(20, np.copy(suppressedLocalMax[0]), num_pixs, edge_pixs, image)
    
    return result_image
  
def hough(binaryImage, resultTheta=1, resultRho=1):
  numRows,numCols = binaryImage.shape
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / resultTheta) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))

  Distance = np.sqrt((numRows - 1)**2 + (numCols - 1)**2)
  q_val = np.ceil(Distance / resultRho)
  new_rho = 2*q_val + 1
  rho = np.linspace(-q_val * resultRho, q_val * resultRho, new_rho)
  Hough_lines = np.zeros((len(rho), len(theta)))
  for rowId in range(numRows):
    for colId in range(numCols):
      if binaryImage[rowId, colId]:
        for thetaId in range(len(theta)):
          rho_val = colId * np.cos(theta[thetaId] * np.pi / 180.0) + \
                    rowId * np.sin(theta[thetaId] * np.pi / 180)
          rho_id1 = np.nonzero(np.abs(rho - rho_val) == np.min(np.abs(rho - rho_val)))[0]
          Hough_lines[rho_id1[0], thetaId] += 1
  return rho, theta, Hough_lines


def gaussian_filter(img_p):
    print("In gaussian")
    img = img_p.copy()
    result_img_g_temp = img.copy()
    l = len(img)
    b = len(img[0])

    for i in range(l):
        for j in range(17,b-17):
            result_img_g_temp[i][j] = ((0.0)*img[i][j-17]) + ((0.0)*img[i][j-16]) + ((0.0)*img[i][j-15]) + ((0.0)*img[i][j-14]) + ((0.0)*img[i][j-13]) + ((0.0)*img[i][j-12]) + ((0.0)*img[i][j-11]) + ((0.0)*img[i][j-10]) + ((0.0)*img[i][j-9]) + ((0.0)*img[i][j-8]) + ((0.0)*img[i][j-7]) + ((0.0)*img[i][j-6]) + ((0.000003)*img[i][j-5]) + ((0.000229)*img[i][j-4]) + ((0.005977)*img[i][j-3]) + ((0.060598)*img[i][j-2]) + ((0.24173)*img[i][j-1]) + ((0.382925)*img[i][j]) + ((0.24173)*img[i][j+1]) + ((0.060598)*img[i][j+2]) + ((0.005977)*img[i][j+3]) + ((0.000229)*img[i][j+4]) + ((0.000003)*img[i][j+5]) + ((0.0)*img[i][j+6]) + ((0.0)*img[i][j+7]) + ((0.0)*img[i][j+8]) + ((0.0)*img[i][j+9]) + ((0.0)*img[i][j+10]) + ((0.0)*img[i][j+11]) + ((0.0)*img[i][j+12]) + ((0.0)*img[i][j+13]) + ((0.0)*img[i][j+14]) + ((0.0)*img[i][j+15]) + ((0.0)*img[i][j+16]) + ((0.0)*img[i][j+17]) 
            

    result_img_g = result_img_g_temp.copy()
    for j in range(b):
        for i in range(17,l-17):
            result_img_g[i][j] =((0.0)*img[i-17][j]) + ((0.0)*img[i-16][j]) + ((0.0)*img[i-15][j]) + ((0.0)*img[i-14][j]) + ((0.0)*img[i-13][j]) + ((0.0)*img[i-12][j]) + ((0.0)*img[i-11][j]) + ((0.0)*img[i-10][j]) + ((0.0)*img[i-9][j]) + ((0.0)*img[i-8][j]) + ((0.0)*img[i-7][j]) + ((0.0)*img[i-6][j]) + ((0.000003)*img[i-5][j]) + ((0.000229)*img[i-4][j]) + ((0.005977)*img[i-3][j]) + ((0.060598)*img[i-2][j]) + ((0.24173)*img[i-1][j]) + ((0.382925)*img[i][j]) + ((0.24173)*img[i+1][j]) + ((0.060598)*img[i+2][j]) + ((0.005977)*img[i+3][j]) + ((0.000229)*img[i+4][j]) + ((0.000003)*img[i+5][j]) + ((0.0)*img[i+6][j]) + ((0.0)*img[i+7][j]) + ((0.0)*img[i+8][j]) + ((0.0)*img[i+9][j]) + ((0.0)*img[i+10][j]) + ((0.0)*img[i+11][j]) + ((0.0)*img[i+12][j]) + ((0.0)*img[i+13][j]) + ((0.0)*img[i+14][j]) + ((0.0)*img[i+15][j]) + ((0.0)*img[i+16][j]) + ((0.0)*img[i+17][j]) 
    
    return result_img_g

def plot_rho_theta_points(target_image, pointPairs):
  image_y_max, image_x_max , channels = np.shape(target_image)
  for i in range(0, len(pointPairs), 1):
    point = pointPairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    left = (0, b)
    right = (image_x_max, image_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((image_y_max - b) / m, image_y_max)

    points_detected = [pt for pt in [left, right, top, bottom] if ifvalidPoint(pt, image_y_max, image_x_max)]
    if len(points_detected) == 2:
      cv2.line(target_image, roundOff_tup(points_detected[0]), roundOff_tup(points_detected[1]), (255, 0, 0), 5)
  return target_image

def ifvalidPoint(pt, y_max, x_max):
  x, y = pt
  if x <= x_max and x >= 0 and y <= y_max and y >= 0:
    return True
  else:
    return False

def roundOff_tup(tuple):
  x,y = [int(round(num)) for num in tuple]
  return (x,y)


def additiveImages(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)
  
def find_values_rho_theta(accMatrix, n, rhos, thetas):
  flat_list = list(set(np.hstack(accMatrix)))
  sorted_flat_list = sorted(flat_list, key = lambda n: -n)
  sorted_coords = [(np.argwhere(accMatrix == acc_value)) for acc_value in sorted_flat_list[0:n]]
  rho_theta_list = []
  x_y_list = []
  for coords_for_val_x in range(0, len(sorted_coords), 1):
    coords_for_val = sorted_coords[coords_for_val_x]
    for i in range(0, len(coords_for_val), 1):
      n,m = coords_for_val[i]
      rho = rhos[n]
      theta = thetas[m]
      rho_theta_list.append([rho, theta])
      x_y_list.append([m, n])
  return [rho_theta_list[0:n], x_y_list]

def imp_poly(img, four_points):
    mask = np.zeros_like(img)   

    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
   
    cv2.fillPoly(mask, four_points, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def imageProcessing(image):
  
  img = np.copy(image)
  
  grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  gaussianImg = gaussian_filter(grey)
  
  cannyImage = canny_detection(grey)
  edges_detected = cannyImage

  xsize = edges_detected.shape[1]
  ysize = edges_detected.shape[0]
  left_bottom = [xsize*0.11, ysize]
  right_bottom = [xsize*0.9, ysize]
  left_top = [xsize*0.35, ysize*0.64]
  right_top = [xsize*0.4, ysize*0.5]

  four_points = np.array([left_bottom, left_top, right_top, right_bottom], np.int)
  mask_img = imp_poly(edges_detected, [four_points])

  rhos, thetas, HoughLines = hough(mask_img)

  rho_theta_values, x_y_values = find_values_rho_theta(HoughLines, 20 , rhos, thetas)
  image_t_lines = image.copy()
  imageHough = plot_rho_theta_points(image_t_lines, rho_theta_values)
  
  result_img = additiveImages(imageHough, img)
  return result_img

output_video = 'finalOutput_1.mp4'
my_clip = VideoFileClip("inputVideo.mp4").subclip(0,3)

video_clip = my_clip.fl_image(imageProcessing)

video_clip.write_videofile(output_video, audio=False)