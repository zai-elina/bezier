import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter



def circle_points(radius, n):
    points=[]
    theta = 2 * np.pi /n
    for i in range(n):
        points.append([int(np.cos(theta * i) * radius),int(np.sin(theta * i) * radius) ])
    even=[]
    odd=[]
    flag=1
    for i in points:
        if (flag%2==0):
            even.append(i)
        else:
            odd.append(i)
        flag+=1
    ev_p=np.array(even, dtype=np.int32)
    od_p=np.array(odd, dtype=np.int32)
    return ev_p,od_p

def bezier_2(img, x0, y0, x1, y1, x2, y2):
    t = np.arange(0, 1, 0.005)
    for i in t:
        x = int((1-i)**2 * x0 + 2*(1-i)*i*x1 + i**2*x2)
        y = int((1-i)**2 * y0 + 2*(1-i)*i*y1 + i**2*y2)
        img[x][y] = [255, 255, 0]

def save_img(img,images):
    frame = plt.imshow(img)
    img = np.full((N, N, 3), 255, dtype=np.uint8)
    images.append([frame])
    return images,img

def even_dots_movement(img,control_points, moving_points):
    for i in range(len(control_points) - 1):
        bezier_2(img, control_points[i, 0], control_points[i, 1], moving_points[i, 0], moving_points[i, 1], control_points[i + 1, 0], control_points[i + 1, 1])
    bezier_2(img, control_points[0, 0], control_points[0, 1], moving_points[-1, 0], moving_points[-1, 1], control_points[-1, 0], control_points[-1, 1])
    control_points=0
    moving_points=0

def odd_dots_movement(img,control_points, moving_points):
    for i in range(len(control_points) - 1):
        bezier_2(img, control_points[i, 0], control_points[i, 1], moving_points[i+1, 0], moving_points[i+1, 1], control_points[i + 1, 0], control_points[i + 1, 1])
    bezier_2(img, control_points[0, 0], control_points[0, 1], moving_points[0, 0], moving_points[0, 1], control_points[-1, 0], control_points[-1, 1])
    control_points=0
    moving_points=0

def get_point1(points,d,flag):
    even=[]
    odd=[]
    for i in range(len(points)):
        if (i+1)%2==1:
            if flag==0:
                odd.append(points[i]* (1 - d) + N//2)
            else:
                odd.append(points[i]* (0.5 + d) + N//2)
        else:
            if flag==0:
                even.append(points[i]* (1 + d) + N//2)
            else:
                even.append(points[i]* (1.5 - d) + N//2)
    ev_p=np.array(even, dtype=np.int32)
    od_p=np.array(odd, dtype=np.int32)
    return od_p,ev_p

def get_point2(points,d,flag):
    even=[]
    odd=[]
    for i in range(len(points)):
        if (i+1)%2==1:
            if flag==2:
                odd.append(points[i]* (1 + d) + N//2)
            else:
                odd.append(points[i]* (1.5 - d) + N//2)
        else:
            if flag==2:
                even.append(points[i]* (1 - d) + N//2)
            else:
                even.append(points[i]* (0.5 + d) + N//2)
    ev_p=np.array(even, dtype=np.int32)
    od_p=np.array(odd, dtype=np.int32)
    return ev_p,od_p

N = 300
radius=100
n=30
fig = plt.figure()
img = np.full((N, N, 3), 255, dtype=np.uint8)
# ev_p, od_p = circle_points(radius, n)
points=[]
theta = 2 * np.pi /n
for i in range(n):
    points.append([int(np.cos(theta * i) * radius),int(np.sin(theta * i) * radius) ])
images=[]
points=np.array(points, dtype=np.int32)
# control_points=np.zeros((15, 2),dtype=np.int32)
# moving_points=np.zeros((15, 2),dtype=np.int32)


for i in range(4):
    d=np.linspace(0, 0.5, 100)
    if i< 2:
        for j in d:
            control_points,moving_points=get_point1(points,j,i)
            even_dots_movement(img,control_points,moving_points)
            images, img =save_img(img,images)
    else:
        for j in d:
            control_points,moving_points=get_point2(points,j,i)
            odd_dots_movement(img,control_points,moving_points)
            images, img =save_img(img,images)



# for i in np.linspace(0, 0.5, 100):
#     control_points=od_p * (1 - i) + N//2
#     moving_points=ev_p * (1 + i) + N//2
#     even_dots_movement(img,control_points,moving_points)
#     images, img =save_img(img,images)

# for i in np.linspace(0, 0.5, 100):
#     control_points=od_p * (0.5 + i) + N//2
#     moving_points=ev_p * (1.5 - i) + N//2
#     even_dots_movement(img,control_points,moving_points)
#     images, img =save_img(img,images)

# for j in np.linspace(0, 0.5, 100):
#     control_points=ev_p * (1 - j) + N//2
#     moving_points=od_p * (1 + j) + N//2
#     odd_dots_movement(img,control_points, moving_points)
#     images, img =save_img(img,images)


# for j in np.linspace(0, 0.5, 100):
#     control_points=ev_p * (0.5 + j) + N//2
#     moving_points=od_p * (1.5 - j) + N//2
#     odd_dots_movement(img,control_points, moving_points)
#     images, img =save_img(img,images)



anim = ArtistAnimation(fig, images, interval=30, blit=True, repeat=False)
anim.save('animation.gif', writer=PillowWriter(fps=30))
