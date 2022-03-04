import numpy as np
import cv2
import scipy.ndimage as ndimage

import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import util

done = False
patch_size = 8

def diffusion_inpaint(mask,img):
    global done
    height,width,d = mask.shape
    cv2.rectangle(mask,(0,0),(width,height),(0,0,0),patch_size-1) # border
    mask = mask[:,:,0]
    mask_orig = np.copy(mask)
    img_orig = np.copy(img)
    cv2.imwrite("data/inpainting_output/mask.png",mask)
    cv2.imwrite("data/inpainting_output/img.png",img)
    count = 0
    num_indices = 0
    while not done:
        count = count + 1
        frontFill = findFrontFill(mask)
        if done:
            break
        reduced_image,reduced_mask = diffusion_fill(frontFill,img,mask)
        reduced_mask_indices = np.where(reduced_mask == 255)
        if len(reduced_mask_indices[0]) == num_indices:
            reduced_image = finish_inpaint(reduced_image,reduced_mask)
            done = True
        num_indices = len(reduced_mask_indices[0])
        img,mask = reduced_image,reduced_mask

    cv2.imwrite("data/inpainting_output/final_image.png",img)
    return img

def finish_inpaint(img,mask):
    last_indices = np.where(mask == 255)
    for i in range(len(last_indices[0])):
        if np.all(img[last_indices[0][i]+1][last_indices[1][i]] != 255):
            img[last_indices[0][i]][last_indices[1][i]] = img[last_indices[0][i]+1][last_indices[1][i]]
        elif np.all(img[last_indices[0][i]-1][last_indices[1][i]] != 255):
            img[last_indices[0][i]][last_indices[1][i]] = img[last_indices[0][i]-1][last_indices[1][i]]
        elif np.all(img[last_indices[0][i]][last_indices[1][i]+1] != 255):
            img[last_indices[0][i]][last_indices[1][i]] = img[last_indices[0][i]][last_indices[1][i]+1]
        elif np.all(img[last_indices[0][i]][last_indices[1][i]-1] != 255):
            img[last_indices[0][i]][last_indices[1][i]] = img[last_indices[0][i]][last_indices[1][i]-1]
    cv2.imwrite("data/inpainting_output/reduced_img.png",img)
    return img


def diffusion_fill(frontFill,img,mask):
    img_w_mask = np.copy(img)
    white_mask = np.where(mask == 255)
    img_w_mask[white_mask] = 255
    for p in range(len(frontFill[0])):
        patch,x_min,x_max,y_min,y_max = get_neighbors(frontFill[1][p],frontFill[0][p],img_w_mask)

        cv2.imwrite("data/inpainting_output/bw_patch.png",patch)
        bw_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        colored_patch_cells = np.where(bw_patch != 255)
        r_channel_avg = 0
        g_channel_avg = 0
        b_channel_avg = 0
        for i in range(patch_size):
            for j in range(patch_size):
                if np.all(patch[i][j] != 255):
                    r_channel_avg = r_channel_avg + patch[i][j][0]
                    g_channel_avg = g_channel_avg + patch[i][j][1]
                    b_channel_avg = b_channel_avg + patch[i][j][2]
        
        valid_pixels = len(colored_patch_cells[0])
        r_channel_avg = r_channel_avg//valid_pixels
        g_channel_avg = g_channel_avg//valid_pixels
        b_channel_avg = b_channel_avg//valid_pixels
        img_w_mask[frontFill[0][p]][frontFill[1][p]] = (r_channel_avg,g_channel_avg,b_channel_avg)
        cv2.imwrite("data/inpainting_output/fill_img.png",img_w_mask)

    mask[frontFill] = 0
    img = img_w_mask
    cv2.imwrite("data/inpainting_output/reduced_mask.png",mask)
    cv2.imwrite("data/inpainting_output/reduced_img.png",img)

    return img,mask

def inpaint(mask,img):
    height,width,d = mask.shape
    cv2.rectangle(mask,(0,0),(width,height),(0,0,0),patch_size-1) # border
    mask = mask[:,:,0]
    mask_orig = np.copy(mask)
    img_orig = np.copy(img)
    cv2.imwrite("data/inpainting_output/mask.png",mask)
    cv2.imwrite("data/inpainting_output/img.png",img)

    while not done:
        frontFill = findFrontFill(mask)
        if done:
            break
        priority,index = computePriority(frontFill,mask,mask_orig,img)
        patched_mask, patched_image = findMatchingPatch(mask,img,priority,frontFill,index)
        mask, img = patched_mask, patched_image
    
    return img

def findFrontFill(mask):
    global done
    height , width = mask.shape
    edges = cv2.Canny(mask,100,200)
    edge_indices = np.where(edges != 0)
    
    if len(edge_indices[0]) == 0:
        edge_indices = np.where(mask == 255)
        if len(edge_indices[0]) == 0:
            done = True
    cv2.imwrite("data/inpainting_output/edges.png",edges)
    
    return edge_indices

def get_neighbors(x,y,patch_img):
    patch_img_orig = np.copy(patch_img)
    height, width, d = patch_img.shape

    x_min = max(0, x - patch_size//2)
    x_max = min(width, x + patch_size//2) 
    y_min = max(0, y - patch_size//2)
    y_max = min(height, y + patch_size//2)

    patch = patch_img[y_min:y_max, x_min:x_max]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = np.zeros((patch_size,patch_size,3))
    return patch,x_min,x_max,y_min,y_max

def computePriority(frontFill_indices,mask,mask_orig,img):
    priority = np.zeros(len(frontFill_indices[0]))
    height,width,d = img.shape
    for p in range(len(frontFill_indices[0])):
        #confidence
        confidence_total = 0
        x,y = frontFill_indices[1][p],frontFill_indices[0][p]
        patch,x_min,x_max,y_min,y_max = get_neighbors(x,y,img)
        for i in range(y_min,y_max):
            for j in range(x_min,x_max):
                if mask[i][j] == 0:
                    confidence_total = confidence_total + 1
        confidence = confidence_total/(patch_size*2)

        #data
        img_copy = np.copy(img)
        bw_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        dx = cv2.Sobel(bw_img,cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(bw_img,cv2.CV_64F,0,1,ksize=5)
        # https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
        magnitude = np.hypot(dx, dy)
        magnitude *= 255.0 / np.max(magnitude)
        cv2.imwrite("data/inpainting_output/gradients.png",magnitude)
        width,height,d = img.shape
        data = magnitude[y][x]/255
        priority[p] = data*confidence

    index = np.argmax(priority)

    return priority,index

def findMatchingPatch(mask,img,priority,frontFill,index):
    img_w_mask = np.copy(img)
    white_mask = np.where(mask == 255)
    img_w_mask[white_mask] = 255

    patch_truth,x_min,x_max,y_min,y_max = get_neighbors(frontFill[1][index],frontFill[0][index],img_w_mask)
    x_min_source,x_max_source,y_min_source,y_max_source = 0,0,0,0
    cv2.imwrite("data/inpainting_output/patch_truth.png",patch_truth)

    #get list
    patch_truth_bw = cv2.cvtColor(patch_truth, cv2.COLOR_RGB2GRAY)
    patch_list = np.where(patch_truth_bw != 255)
    smallest_error = np.inf
    for i in range(patch_size//2,img.shape[0]-patch_size//2):
        for j in range(patch_size//2,img.shape[1]-patch_size//2):
            patch,x_min2,x_max2,y_min2,y_max2 = get_neighbors(j,i,img_w_mask)
            patch_truth = cv2.resize(patch_truth,(patch_size,patch_size))

            # calculate error
            error = np.sqrt(np.mean((patch_truth[patch_list] - patch[patch_list] )**2))
            if error < smallest_error:
                if i < frontFill[0][index] - patch_size//2 or i > frontFill[0][index] + patch_size//2:
                    if j < frontFill[1][index] - patch_size//2 or j > frontFill[1][index] + patch_size//2:
                        if j != frontFill[1][index] or i != frontFill[0][index]:
                            x_min_source,x_max_source,y_min_source,y_max_source = x_min2,x_max2,y_min2,y_max2
                            smallest_error = error

    location_patch_replace = np.copy(img)
    cv2.rectangle(location_patch_replace, (frontFill[1][index]-patch_size//2,frontFill[0][index]+patch_size//2),(frontFill[1][index]+patch_size//2,frontFill[0][index]-patch_size//2),(255,255,255))
    cv2.rectangle(location_patch_replace, (x_min_source,y_max_source),(x_max_source,y_min_source),(255,0,0))
    cv2.imwrite("data/inpainting_output/location_patch_replace.png",location_patch_replace)

    patched_image = np.copy(img)
    patched_mask = np.copy(mask)
    
    cv2.imwrite("data/inpainting_output/img_patch.png",img[y_min_source:y_max_source,x_min_source:x_max_source])
    cv2.imwrite("data/inpainting_output/copy_patch.png",patched_image[y_min:y_max,x_min:x_max])
    patched_image[y_min:y_max,x_min:x_max] = img[y_min_source:y_max_source,x_min_source:x_max_source]
    patched_mask[y_min:y_max,x_min:x_max] = 0

    cv2.imwrite("data/inpainting_output/patched_mask.png",patched_mask)
    cv2.imwrite("data/inpainting_output/patched_image.png",patched_image)

    return patched_mask, patched_image

#alias
def inpaint_diffusion(mask,img):
    matte = diffusion_inpaint(mask,img)
    return matte

#alias
def inpaint_patch_match(mask,img):
    matte = inpaint(mask,img)
    return matte

if __name__ == "__main__":
    mask = cv2.imread("data/output/mask.png")
    img = cv2.imread("data/output/img.png")
    
    inpaint(mask,img)
    # diffusion_inpaint(mask,img)