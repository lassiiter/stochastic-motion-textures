import numpy as np
import numpy.linalg
from skimage import data
from skimage.restoration import inpaint
import cv2
import matplotlib.pyplot as plt
from pymatting import cutout
import random
from matting import *
from inpaint import *

mode = True  # if True, draw using white, false draw grey
annotate = False  # true if mouse is pressed
brush_size = 25
num_frames = 100
x1,y1,x2,y2 = 0,0,0,0
count = 1
flip = False
count = 0


def cutout_image_naive(tri_map,path,img,orig_img):
   
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    ret,mask = cv2.threshold(tri_map,1,255,cv2.THRESH_BINARY)
    mask = np.uint8(mask)
    
    cv2.imwrite("data/output/tri_map.png",tri_map)
    cv2.imwrite("data/output/img.png",img)

    ###### Attempted to implement https://grail.cs.washington.edu/projects/digital-matting/papers/cvpr2001.pdf. Struggled with implementation. See bayesian_matting
    # img = cv2.imread('data/lemur.png')
    # tri_map = cv2.imread('data/lemur_trimap.png')
    # matte = bayesian_matte(tri_map,img)

    ###### self implementation
    mask = naive_matte(tri_map,orig_img)
    mask = np.uint8(mask)
    kernel = np.ones((5,5),np.uint8)
    mask_dilated = cv2.dilate(mask,kernel)
    cv2.imwrite("data/output/mask.png",mask_dilated)
    cv2.imwrite("data/output/mask_orig.png",mask_dilated)

    

    background_plate = cv2.inpaint(orig_img,mask,5,cv2.INPAINT_TELEA)

    cv2.imwrite("data/output/background.png",background_plate)
    rgba_background_plate = cv2.cvtColor(background_plate, cv2.COLOR_RGB2RGBA)

    mask = cv2.erode(mask,kernel)
    matte = cv2.bitwise_and(img,img,mask=mask)
    cv2.imwrite("data/output/cutout.png",matte)
    return rgba_background_plate,matte

def cutout_image(tri_map,path,img,orig_img):
    kernel = np.ones((5,5),np.uint8)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    ret,mask = cv2.threshold(tri_map,1,255,cv2.THRESH_BINARY)
    mask = np.uint8(mask)

    cv2.imwrite("data/output/tri_map.png",tri_map)
    cv2.imwrite("data/output/img.png",img)

    # pymatting cut library
    cutout(path,"data/output/tri_map.png","data/output/cutout.png")
    matte = cv2.imread("data/output/cutout.png",cv2.IMREAD_UNCHANGED)
    ret,mask = cv2.threshold(matte[:,:,3],50,255,cv2.THRESH_BINARY)
    mask_dilated = cv2.dilate(mask,kernel)
    cv2.imwrite("data/output/mask.png",mask_dilated)
    cv2.imwrite("data/output/mask_orig.png",mask_dilated)
    cv2.imwrite("data/output/matte_orig.png",matte)

    ######## fastest solution from library
    background_plate = cv2.inpaint(orig_img,mask_dilated,5,cv2.INPAINT_TELEA)

    ######## diffusion fill, faster than patch match but slow
    # mask_dilated = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2RGB)
    # background_plate = inpaint_diffusion(mask_dilated,orig_img)

    ######## patch match fill, very very slow
    # mask_dilated = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2RGB)
    # background_plate = inpaint_patch_match(mask_dilated,orig_img)

    cv2.imwrite("data/output/background.png",background_plate)
    rgba_background_plate = cv2.cvtColor(background_plate, cv2.COLOR_RGB2RGBA)

    cv2.imwrite("data/output/cutout.png",matte)
    return rgba_background_plate,matte

def draw(event, x, y, flags, param):
    # mouse callback function
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
    global brush_size,annotate, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        annotate = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if annotate == True:
            if mode:
                cv2.circle(img, (x, y), brush_size, (255, 255, 255), -1)
                cv2.circle(tri_map, (x, y), brush_size, (255), -1)
                cv2.circle(tri_map_preview, (x, y), brush_size, (1), -1)
            else:
                cv2.circle(img, (x, y), brush_size, (125, 125, 125), -1)
                cv2.circle(tri_map, (x, y), brush_size, (125), -1)
                cv2.circle(tri_map_preview, (x, y), brush_size, (.5), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        annotate = False
        if mode:
            cv2.circle(img, (x, y), brush_size, (255, 255, 255), -1)
            cv2.circle(tri_map, (x, y), brush_size, (255), -1)
            cv2.circle(tri_map_preview, (x, y), brush_size, (1), -1)

        else:
            cv2.circle(img, (x, y), brush_size, (125, 125, 125), -1)
            cv2.circle(tri_map, (x, y), brush_size, (125), -1)
            cv2.circle(tri_map_preview, (x, y), brush_size, (.5), -1)

def get_points(event, x, y, flags, param):
    # mouse callback function
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
    global x1,y1,x2,y2,count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        count = count + 1
        if count % 2 == 0:
            x1,y1 = x,y
            print("x1,y1: " + str(x1),str(y1))
        else:
            x2,y2 = x,y
            print("x2,y2: " + str(x2),str(y2))
        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 0), thickness=5)

def get_trimap():
    global brush_size,annotate, mode
    cv2.namedWindow('image')
    cv2.namedWindow('tri_map')
    cv2.setMouseCallback('image', draw)
    cv2.setMouseCallback('tri_map', draw)
    print("Mask area starting closest: x = mode, s = save, [ = shrink brush, ] = grow brush")
    while(1):
        cv2.imshow('image', img)
        cv2.imshow('tri_map', tri_map_preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('x'):
            mode = not mode
        if k == ord('s'):
            break
        if k == ord(']'):
            brush_size=brush_size +5
        if k == ord('['):
            brush_size=brush_size -5
        elif k == 27:
            break
    cv2.destroyAllWindows()
    return img,tri_map,path

def get_motion_type():
    motion_type = ""
    print("Label layer: b = boat, p = plant, w = water, c = cloud, r = rock (stationary)")
    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            motion_type = "boat"
            break
        if k == ord('p'):
            motion_type = "plant"
            break
        if k == ord('w'):
            motion_type = "water"
            break
        if k == ord('c'):
            motion_type = "cloud"
            break
        if k == ord('r'):
                motion_type = "rock"
                break
        elif k == 27:
            break
    cv2.destroyAllWindows()
    return motion_type

def get_line_segment():
    global x1,y1,x2,y2
    print("Draw a line under the boat, or the length of the plant, by clicking two points. s = save")
    print("For plants start at the tip, x1,y1 should be at the tip")

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_points)
    while(1):
        cv2.imshow('image', img_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print(x1,y1,x2,y2)
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()
    return motion_type

def generate_displacment(layers,i):
    final_image = np.copy(layers[-1][0])

    for layer in layers:
        background = layer[0]
        matte = layer[1]
        motion_type = layer[2]

        r,g,b,mask = cv2.split(layer[1])
        cv2.imwrite("data/output/last_mask.png",mask)
        invertedMask = 255 - mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
        invertedMask = cv2.cvtColor(invertedMask, cv2.COLOR_RGB2RGBA)

        if motion_type == "cloud":
            final_image = cloud_displacement(final_image,mask,matte,invertedMask,i)
        elif motion_type == "plant":
            final_image = plant_displacement(final_image,mask,matte,invertedMask,i)
        elif motion_type == "water":
            final_image = water_displacement(final_image,mask,matte,invertedMask,i)
        elif motion_type == "rock":
            final_image = final_image
        elif motion_type == "boat":
            final_image = boat_displacement(final_image,mask,matte,invertedMask,i)
    return final_image

def water_displacement(final_image,mask,matte,invertedMask,t):
    mask = mask[:,:,0]
    mask_indices = np.where(mask == 255)
    step = np.sin(t)
    for i in range(len(mask_indices[0])):
        for j in range(len(mask_indices[1])):
            matte[i][j] = np.roll(matte[i][j], step,axis=1)
            mask = np.roll(mask, step,axis=1)
            invertedMask = np.roll(invertedMask, step,axis=1)

    final_image = final_image * (invertedMask/255) + matte *(mask/255)
    return final_image

def cloud_displacement(final_image,mask,matte,invertedMask,i):
    step = i//2
    matte = np.roll(matte, step,axis=1)
    mask = np.roll(mask, step,axis=1)
    invertedMask = np.roll(invertedMask, step,axis=1)
    final_image = final_image * (invertedMask/255) + matte *(mask/255)
    return final_image

def boat_displacement(final_image,mask,matte,invertedMask,i):
    global flip
    height, width, d = final_image.shape
    step = i%3 + 1
    step = step/2
    step = np.uint8(step)
    final_image_copy = np.copy(final_image)
    mask_indices = np.where(mask == 255)
    center = (mask_indices[1][len(mask_indices[0])//2],mask_indices[0][len(mask_indices[0])//2])
    if i % 3 == 0:
        flip = not flip
    if flip:
        rotation_step = (((step-1)/2)/10) + (1/random.randint(6,10))
        rot_mat = cv2.getRotationMatrix2D(center, rotation_step, 1.0)
        matte = cv2.warpAffine(matte, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        invertedMask = cv2.warpAffine(invertedMask, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)

        matte = np.roll(matte, step,axis=0)
        mask = np.roll(mask, step,axis=0)
        invertedMask = np.roll(invertedMask, step,axis=0)
    else:
        step = 3-step
        rotation_step = (((step-1)/2)/10) + (1/random.randint(6,10))
        rot_mat = cv2.getRotationMatrix2D(center, rotation_step, 1.0)
        matte = cv2.warpAffine(matte, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)
        invertedMask = cv2.warpAffine(invertedMask, rot_mat, final_image.shape[1::-1], flags=cv2.INTER_LINEAR)

        matte = np.roll(matte,step,axis=0)
        mask = np.roll(mask, step,axis=0)
        invertedMask = np.roll(invertedMask,step,axis=0)
    final_image = final_image * (invertedMask/255) + matte *(mask/255)
    crop_amount = 10
    return final_image[0+crop_amount:height-crop_amount,0+crop_amount:width-crop_amount]

def plant_displacement(final_image,mask,matte,invertedMask,t):
    global x1,x2,y1,y2,count
    matte = cv2.imread("data/output/matte_orig.png",cv2.IMREAD_UNCHANGED)
    r,g,b,mask = cv2.split(matte)
    invertedMask = 255 - mask
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
    invertedMask = cv2.cvtColor(invertedMask, cv2.COLOR_RGB2RGBA)
    
    mask_indices = np.where(mask != 0)
    wind_force = .7
    natural_frequency = [1.2]
    mass = 2
    velocity_dampening = .05

    #https://stackoverflow.com/questions/57086256/finding-width-and-height-of-concave-curved-shaped-blob
    mask_2d = mask[:,:,0]
    M = cv2.moments(mask_2d)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    column_pixels = cv2.countNonZero(mask_2d[:, cX])
    mask_indices = np.nonzero(mask_2d[:, cX])
   
    max_distance = np.sqrt((y2-y1)**2)
    phase_shift = np.arctan((natural_frequency[0])*velocity_dampening/1+ ((natural_frequency[0]**2) - (natural_frequency[0]**2))+1)
    top = wind_force**(2*np.pi*phase_shift)
    bottom = 2*np.pi*mass*np.sqrt((numpy.fft.ifft(natural_frequency)**2 - natural_frequency[0]**2)**2 + velocity_dampening**2*(numpy.fft.ifft(natural_frequency)**2))
    D_tip = top/bottom
    d_tip = np.fft.ifft(D_tip)
    
    # noise applied to each row
    gaussian_r = np.random.normal(scale=.1)
    # larger sway numbers actually reduce sway
    sway = random.uniform(1,2)

    for i in range(column_pixels):
        y = mask_indices[0][i]
        distance = np.sqrt((y-y1)**2)
        u = distance/max_distance
        displacement_hold = (((1/3*u)**4) - ((1/3*u)**4) + ((2*u)**4))
        displacement = (np.exp(u/sway * (np.sin((1/2)*t) + gaussian_r)))
        displacement = displacement* displacement_hold
        displacement = np.uint8(displacement)

        mask[y][:] = np.roll(mask[y][:], displacement,axis=0)
        matte[y][:] = np.roll(matte[y][:], displacement,axis=0)
        invertedMask[y][:] = np.roll(invertedMask[y][:], displacement,axis=0)
       

    final_image = final_image * (invertedMask/255) + matte *(mask/255)
    
    return final_image

if __name__ == "__main__":
    path = 'data/3.png'
    img = cv2.imread(path)
    img_copy = np.copy(img)
    orig_img = np.copy(img)
    orig2_img = np.copy(img)
    
    layers = []
    num_layers = int(input("How Many Layers do you want to process: "))
    #layering + matting
    for i in range(num_layers):
        tri_map = np.zeros((img.shape[0],img.shape[1]), np.float64)
        tri_map_preview = np.zeros((img.shape[0],img.shape[1]), np.float64)
        motion_type = get_motion_type()
        img,tri_map,path = get_trimap()
        background,matte = cutout_image(tri_map,path,orig2_img,orig_img)
        img = background
        if motion_type == "plant" or motion_type == "boat":
            line = get_line_segment()
            layers.append((background,matte,motion_type,line))
        else:
            layers.append((background,matte,motion_type))
        background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
        orig2_img, orig_img = background,background
    
    #render
    for i in range(num_frames):
        final_image = generate_displacment(layers,i)
        cv2.imwrite("data/image_sequence/image" + str(i) +".png",final_image)
        
    