import numpy as np
import cv2
import scipy.ndimage as ndimage

x1,y1 = 0,0

def foreground_background_unknown(tri_map):
    cv2.imwrite("bayesian_test/tri_map.png",tri_map)
    
    foreground = np.copy(tri_map)
    background = np.zeros((tri_map.shape))
    unknown = np.zeros((tri_map.shape))

    ret,foreground = cv2.threshold(tri_map,254,255,cv2.THRESH_BINARY)
    indices = np.where(tri_map==0)
    background[indices[0], indices[1], :] = [255, 255, 255]
    unknown = 255 - (background + foreground)

    cv2.imwrite("bayesian_test/foreground.png",foreground)
    cv2.imwrite("bayesian_test/background.png",background)
    cv2.imwrite("bayesian_test/unknown.png",unknown)

    return foreground,background,unknown

def get_gaussian_kernel(neigborhood_size):
    #"Second, we use a spatial Gaussian falloff gi with σ = 8 t"
    gauss_kernel_2d = cv2.getGaussianKernel(neigborhood_size,sigma=8)
    gauss_kernel = np.dot(gauss_kernel_2d,gauss_kernel_2d.T)
    return gauss_kernel

def get_neighborhood(img,x,y,neigborhood_size):
    ##  doesnt account for edges
    return img[y-neigborhood_size//2:y+neigborhood_size//2,x-neigborhood_size//2:x+neigborhood_size//2]

def get_cluster(weight,img):
    #Orchard and Bouman Cluster???
    return 0,1

def bayesian_matte(tri_map,img):
    neigborhood_size = 25
    foreground,background,unknown = foreground_background_unknown(tri_map)
    unknown = unknown[:,:,0] #3d to 2d
    unknown_indices = np.where(unknown!=0)
    guasssian_weights = get_gaussian_kernel(neigborhood_size)

    for i in range(len(unknown_indices[0])):
        y,x = unknown_indices[0][i],unknown_indices[1][i]

        alpha_neighborhood = get_neighborhood(unknown,x,y,neigborhood_size)

        background_neighborhood = get_neighborhood(background,x,y,neigborhood_size)
        background_weights = (alpha_neighborhood**2 * guasssian_weights)
        background_indices = np.where(background_weights > 0)
        background_weights = background_weights[background_indices]

        foreground_neighborhood = get_neighborhood(foreground,x,y,neigborhood_size)
        foreground_weights = ((1-alpha_neighborhood)**2 * guasssian_weights)
        foreground_indices = np.where(foreground_weights > 0)
        foreground_weights = foreground_weights[foreground_indices]

        foreground_mean_color, foreground_covariance_matrix = get_cluster(foreground_weights,foreground_neighborhood)
        background_mean_color, background_covariance_matrix = get_cluster(background_weights,background_neighborhood)

    return 0


def naive_matte(tri_map,img):
    img_copy = np.copy(img)
    mask = np.zeros(tri_map.shape)
    mask.fill(255)
    background_color = get_background(img)
    unknown_indices = np.where(tri_map == 125)
    known_indices = np.where(tri_map == 255)
    threshold = 10
    for i in range(len(unknown_indices[0])):
        x,y = unknown_indices[0][i],unknown_indices[1][i]
        if img[x][y][0] < (background_color[0] + threshold) and img[x][y][0] > (background_color[0] - threshold):
            if img[x][y][1] < (background_color[1] + threshold) and img[x][y][1] > (background_color[1] - threshold):
                if img[x][y][2] < (background_color[2] + threshold) and img[x][y][2] > (background_color[2] - threshold):
                    mask[x][y] = 255
                else:
                    mask[x][y] = 0
            else:
                mask[x][y] = 0
        else:
            mask[x][y] = 0
    mask = 255 - mask
    mask[known_indices] = 255
    return mask

def get_xy(event, x, y, flags, param):
    global x1,y1
    if event == cv2.EVENT_LBUTTONDOWN:
        x1,y1 = x,y
        print("Color Chosen!")

def get_background(img):
    global x1,y1
    img_copy = np.copy(img)
    print("Click closest background color. s = save")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_xy)
    while(1):
        cv2.imshow('image', img_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            break
        elif k == 27:
            break
    cv2.destroyAllWindows()
    return img_copy[x1][y1]

if __name__ == "__main__":
    mask = cv2.imread("data/output/mask.png")
    img = cv2.imread("data/output/img.png")
    tri_map = cv2.imread("data/output/tri_map.png")
    mask = naive_matte(tri_map,img)
    cv2.imwrite("data/bayesian_matte_output/naive_matte_mask.png",mask)
