# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 17:46:01 2020 @author: Alexm 

# particles is a list with a few particles in it
# particlez = particles[say x=2]
# particleDISP_SIZE => [mass, velocity_x, veloctiy_y, location_x, location_y]


Works OK but radius seems small when its large and trying to suck things in 
Also objects accelerate to fast never seen and planets or twin stars so gravities up a bit


"""
import random
import pygame
from itertools import count
clock = pygame.time.Clock()

FRAME_RATE = 400
DISP_SIZE  = (800,600)
WHITE      = (255,255,255)
BLACK      = (0  ,0  ,0  )

max_nparticle = 15
max_velco     = 1
max_mass      = 40
acc_cutoff    = 10
G = 0.3

WRAP                     = False
DEFLECT                  = True
JOIN_IF_NEAR             = False
SLOW_DOWN_HIGH_VELOCITIY = True
LIMIT_TO_CENTER_QUATER   = True

if SLOW_DOWN_HIGH_VELOCITIY:
#    factor      = 0.98
#    speed_limit = 2.5
    factor      = 0.92
    speed_limit = 0.5
    



def create_particles(velco=max_velco,mass=max_mass):
    box_starting = [(0, DISP_SIZE[0]) , (0, DISP_SIZE[1])]
    if LIMIT_TO_CENTER_QUATER:
       box_starting = [(int(DISP_SIZE[0]/4),int(DISP_SIZE[0]*3/4)) , (int(DISP_SIZE[1]/4),int(DISP_SIZE[1]*3/4))]
    
    nparticles = random.randint(2,max_nparticle)
    nparticles = 4 # instead of a random number pick an exact
    
    particles = [ [random.uniform(a, b) for a,b in [(0,mass),(-velco,velco),(-velco,velco),box_starting[0],box_starting[1]]] for _ in range(nparticles)]
    print(f"The number of partilces is {nparticles}")
    
    if True: # Add an extra particle that wont move
        particles.append([100,0,0,DISP_SIZE[0]/2,DISP_SIZE[1]/2, None])
    return particles

def update_particles(particles,velcoity_update2,WRAP=WRAP,DEFLECT=DEFLECT,a_delta=0.1):
    for i in range(len(particles)):
        if len(particles[i])==5:
            # Update the particles values # appart from mass
            particles[i][3]+= a_delta*particles[i][1] # add a velocity change to the x location
            particles[i][4]+= a_delta*particles[i][2]
            particles[i][1]-= velcoity_update2[i][0] # add some acceleration to the veclocity
            particles[i][2]-= velcoity_update2[i][1]    
            
        if WRAP:
           particles[i][3] = particles[i][3] %(DISP_SIZE[0])
           particles[i][4] = particles[i][4] %(DISP_SIZE[1])
           
        if DEFLECT:
           if not DISP_SIZE[0]> particles[i][3]-velcoity_update2[i][0]>0:
              particles[i][1] *=-1
           if not DISP_SIZE[1]> particles[i][4]-velcoity_update2[i][1]>0:
              particles[i][2] *=-1 
              
        if SLOW_DOWN_HIGH_VELOCITIY:
            particles[i][1] =  particles[i][1] if  speed_limit> particles[i][1]> -speed_limit else particles[i][1] *factor
            particles[i][2] =  particles[i][2] if  speed_limit> particles[i][2]> -speed_limit else particles[i][2] *factor
    return particles






def magn(val):
    return 1 if val>0 else -1

def div(a,b):
    return a/b if b!=0 else 0

def transpose(mat):
    return list(zip(*mat))


def find_grav_vector(locs1,locs2,multi=1):
    differences = [a-b for a,b in zip(locs1,locs2)]
    distance3 = sum([i**2 for i in differences])**(3/2)
    if abs(distance3)<0.3:
        return [0 for _ in locs1]
    return [G*multi*d/distance3 for d in differences]

#f= m1m2 / (((d1x-d2x)**2 + (d1y-d2y)**2)*(1/2))

out = find_grav_vector([2,0],[0,0])


def remove_and_join_particles(remove_list,particles):    
    for ind_from, ind_to in reversed(remove_list):
     
        particle_to,particle_from = particles[ind_to], particles[ind_from]
        if particle_to is not None and particle_from is not None:
            m1,m2 =  particle_to[0], particle_from[0]
            m12 = m1+m2
            for n in [1,2,3,4]:
                particles[ind_to][n] = (m1/m12)*particle_to[n]  +  (m2/m12)*particle_from[n] 
            particles[ind_to][0]=m12
            particles[ind_from] = None
            if [len(particle_from),len(particle_to)]==[6,5]:
               particles[ind_to] =  particles[ind_to]+ [None,]
    particles =[p for p in particles if p is not None]
    return particles

##########################################################
    
def draw(win,particles):
    win.fill(BLACK)
    for particle in particles:
       if len(particle)==5:
          pygame.draw.circle(win,WHITE,(int(particle[3]),int(particle[4])),int(particle[0]**(1/2)))
       #Added so you can add a object that does not move
       else:
          pygame.draw.circle(win,(0,50,50),(int(particle[3]),int(particle[4])),int(particle[0]**(1/2)))
    pygame.display.update()
##################################################################################################### 
def init(gamename="Particle Gravity Simulation",DISP_SIZE=DISP_SIZE):
    pygame.init()
    win = pygame.display.set_mode(DISP_SIZE)
    pygame.display.set_caption(gamename)
    return win
    
def should_the_game_be_quitted():
    clock.tick(FRAME_RATE) # this slows it down
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True    

play_game = True
if play_game:
    for counter in count():
        if counter == 0:
           win = init() 
        else:
           draw(win,particles)
        if should_the_game_be_quitted():
           break 
##################################################################################### PYGAME END
        if counter==0:
            particles = create_particles()
        if JOIN_IF_NEAR:       
            # some of this may useful when Ive got the equations correct
            particle_difference_location = [[  [p1[3]-p2[3],p1[4]-p2[4]] for p2 in particles ] for p1  in particles]
            
            particle_distance_matrix     = [[ sum([r**2 for r in row])**(1/2) for row in col] for col in particle_difference_location]
            # escapevel = mass/distance#  Rg = 2GM/c2
            escape_vel                   = [[  div(p_mass[0],distance_row) for distance_row in col ] for p_mass,col in  zip(particles,particle_distance_matrix)]
            remove_list = [[  (icol,irow) if (2*row)**(1/3)>3 else None  for irow,row in enumerate(col)] for icol,col in enumerate(escape_vel)]
            remove_list = [a for b in remove_list for a in b if a is not None]
            particles = remove_and_join_particles(remove_list,particles)    
        
        parmat = [[  find_grav_vector([p1[3],p1[4]],[p2[3],p2[4]],multi=p2[0]) for p2 in particles] for p1 in particles]

        velcoity_update = [transpose(mat) for mat in parmat]
        velcoity_update2 = [[ (abs(sum(n))**(1/2)) *sum(n) for n in mat ] for mat in velcoity_update]
        
        particles = update_particles(particles,velcoity_update2)



    
    
    
    
    