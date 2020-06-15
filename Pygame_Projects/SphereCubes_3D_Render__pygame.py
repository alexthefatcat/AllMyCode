# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:05:47 2020 @author: Alexm
No not me:-  XXDLCENERGYXX
"""
import pygame, sys, os, math

"""


wsad are movement
escape to quie
qe zoom


key presses to change rotation
sun: white ball with shadows
spacebar:click to create new squares in areans

unit_circle_points

"""





def rotate2d(pos,rad): 
    x,y = pos
    s,c = math.sin(rad),math.cos(rad)
    return x*c-y*s,y*c+x*s


class Cam:
    def __init__(self,pos=(0,0,0),rot=(0,0),drot=(0,0)):
        self.pos = list(pos)
        self.rot = list(rot)
        self.drot = list(drot)

    def events(self,event):
        if event.type == pygame.MOUSEMOTION:
            x,y = event.rel
            x/=200
            y/=200
            self.rot[0]+=y
            self.rot[1]+=x

    def update(self,dt,key):
        s = dt*4

        if key[pygame.K_r]: self.drot[0]+=0.2
        if key[pygame.K_f]: self.drot[0]-=0.2
        if key[pygame.K_r] and key[pygame.K_f]:
            self.drot[0]=0
        self.rot[0]+= math.pi*dt*self.drot[0]

        if key[pygame.K_q]: self.pos[1]+=s
        if key[pygame.K_e]: self.pos[1]-=s

        x,y = s*math.sin(self.rot[1]),s*math.cos(self.rot[1])
        if key[pygame.K_w]: self.pos[0]+=x; self.pos[2]+=y
        if key[pygame.K_s]: self.pos[0]-=x; self.pos[2]-=y
        if key[pygame.K_a]: self.pos[0]-=y; self.pos[2]+=x
        if key[pygame.K_d]: self.pos[0]+=y; self.pos[2]-=x






# ------------------------- Generating Sphere --------------------------- #
def create_vertices_and_points_of_sphere(longitude_A=9,latitude_B=18):
    #A,B = 9,18 # 
    A,B = longitude_A,latitude_B
    
    points = [(0,-1,0)]
    for zRot in range(A,180,A):
        X,y = rotate2d((0,-1),zRot/180*math.pi)
        for yRot in range(0,360,B):
            z,x = rotate2d((0,X),yRot/180*math.pi)
            points+=[(x,y,z)]
    points+=[(0,1,0)]
    
    a = len(range(A,180,A)); b = len(range(0,360,B))
    n = len(points)-1; n2 = b*(a-1)
    
    po = []
    for i in range(1,b+1):
        if i==b: po+=[(0,i,1)]
        else: po+=[(0,i,i+1)]
    for j in range(0,(a-1)*b,b):
        for i in range(1,b+1):
            if i==b: po+=[(i+j,i+b+j,i+1+j,1+j)]
            else: po+=[(i+j,i+b+j,i+b+1+j,i+1+j)]
    for i in range(1,b+1):
        if i==b: po+=[(n,i+n2,1+n2)]
        else: po+=[(n,i+n2,i+1+n2)]
    return points,po

points,po = create_vertices_and_points_of_sphere()

class Sphere:
    vertices = points
    faces    = po
    colors = (255,0,0),(255,255,0),(0,255,255)
    shape = "Sphere"

    def __init__(self,pos=(0,0,0)):
        x,y,z = pos
        self.verts = [(x+X/2,y+Y/2,z+Z/2) for X,Y,Z in self.vertices]

# ------------------------- Generated Sphere --------------------------- #
class Cube:
    vertices = (-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(0.5,0.5,-0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)
    faces = (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)
    colors = (255,0,0),(255,128,0),(255,255,0),(255,255,255),(0,0,255),(0,255,0)
    shape = "Cube"
    def __init__(self,pos=(0,0,0),momentum=1): 
        self.momentum = momentum
        self.pos = pos
        x,y,z = pos
        self.verts=[(x+X,y+Y,z+Z)  for X,Y,Z in self.vertices]
    def move_block_a_little_in_x(self):
        x,y,z = self.pos
        if abs(x)>3:
            self.momentum *= -1
        x    +=  0.2*self.momentum
        # create a new cube and knick some of the information
        temp = Cube((x,y,z))
        self.verts = temp.verts
        self.pos = temp.pos
        
        
        

# Create the objects to be rendered
spheres = [Sphere((x,0,z)) for x,z   in ((-1,0),(0,0),(1,0))]
cubes   = [Cube((x,y,z))   for x,y,z in ((0,1,0),(0,-1,0))]
all_shapes = spheres + cubes




pygame.init()
w,h = 800,600; cx,cy = w//2,h//2; fov = min(w,h)
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.display.set_caption('3D Graphics')
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()
cam = Cam((0,0,0),(math.pi/2,0))
pygame.event.get(); pygame.mouse.get_rel()
pygame.mouse.set_visible(0); pygame.event.set_grab(1)






rot = 0
rotate_camera = True

while True:
    dt = clock.tick()/5000
    
    if rotate_camera:
       rot+=math.pi*dt*0.5

    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
        cam.events(event)

    screen.fill((128,128,255))
    
    #hack I added
    all_shapes[-1].move_block_a_little_in_x()


    # need to go though all objects, get all faces...

    face_list = []; face_color = []; depth = [] # stores all face data
    
    for obj in all_shapes:

        vert_list = []; screen_coords = []
        for x,y,z in obj.verts:

            x,z = rotate2d((x,z),rot)

            x-=cam.pos[0]; y-=cam.pos[1]; z-=cam.pos[2]
            x,z = rotate2d((x,z),cam.rot[1])
            y,z = rotate2d((y,z),cam.rot[0])
            vert_list += [(x,y,z)]
            f = fov/z if z else fov; x,y = x*f,y*f
            screen_coords+=[(cx+int(x),cy+int(y))]


        for f in range(len(obj.faces)):
            face = obj.faces[f]

            on_screen = False
            for i in face:
                x,y = screen_coords[i]
                if vert_list[i][2]>0 and x>0 and x<w and y>0 and y<h: on_screen = True; break

            if on_screen:
                coords = [screen_coords[i] for i in face]
                face_list += [coords]
                face_color += [obj.colors[f%len(obj.colors)]]
                depth += [sum(vert_list[i][2] for i in face)/len(face)]



    # final drawing part, all faces from all objects
    order = sorted(range(len(face_list)),key=lambda i:depth[i],reverse=1)
    for i in order:
        try: pygame.draw.polygon(screen,face_color[i],face_list[i])
        except: pass

    pygame.display.flip()
    
    key = pygame.key.get_pressed()
    cam.update(dt,key)
    
    
    
    
    
