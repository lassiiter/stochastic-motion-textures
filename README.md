# Animating Pictures with Stochastic Motion Textures
Enhancing still pictures with subtly animated motion

## Implementation
This project implements Animating Pictures with Stochastic Motion Textures (2005) within this implementation includes (2004) Region Filling and Object Removal by
Exemplar-Based Image Inpainting and A Bayesian Approach to Digital Matting aswell as A Bayesian Approach to Digital Matting (2001). In this implementation youll find simplified implementations of these in order to have an appropriate output.  

![Comp Photography Pipeline 2drawio](https://user-images.githubusercontent.com/50963416/156677011-5b9d15d3-8b33-4c96-9366-1942dfd663eb.png)


https://grail.cs.washington.edu/projects/StochasticMotionTextures/  
https://grail.cs.washington.edu/projects/digital-matting/papers/cvpr2001.pdf  
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_tip2004.pdf  

## Matting 
After manual user annotation and labeling of location and type, the algorithm solves the grey areas of the image to create a mask 
![5-ORIG](https://user-images.githubusercontent.com/50963416/156677777-b933776c-362d-45ea-9d82-4c1fdb29dfb9.png)
![5-TRIMAP](https://user-images.githubusercontent.com/50963416/156677776-4259d52c-0344-4db4-98d0-82140ad85c4d.png)
![5-MATTE](https://user-images.githubusercontent.com/50963416/156677774-fc44911c-ca5d-43a8-b5d7-460e54b7e648.png)

## Inpainting Process  
![edges](https://user-images.githubusercontent.com/50963416/156676042-46bb6432-da87-4190-9cba-f865d1f7f2d1.png)
![location_patch_replace](https://user-images.githubusercontent.com/50963416/156676037-0757a667-47dd-4f38-afcf-f4a8c477562d.png)
![img](https://user-images.githubusercontent.com/50963416/156676045-d01d04ef-9069-4400-9940-74946e94d1d7.png)
![patched_image](https://user-images.githubusercontent.com/50963416/156676038-2e43de2b-b88e-4a6b-b6dc-2d14ac0b4050.png)

## Final Result Examples
![input_1](https://user-images.githubusercontent.com/50963416/156674937-b94d0e7c-9bc3-4163-9b71-93ae0f335295.png)
![result_1](https://user-images.githubusercontent.com/50963416/156674947-33ec5ede-0c4d-4786-a9ea-79b56f67e8df.gif)

![result_1](https://user-images.githubusercontent.com/50963416/156675762-1110905e-735f-4a12-8ad8-19cb32059178.gif)
