# -*- coding: utf-8 -*-
"""
DMP integration in cartisian or joint cordinates

Created on Wed Oct 18 09:36:17 2017

@author: RokPahic

VERSION 1.2
"""

#imports
import numpy as np



class DMP(object):



    def __init__(self,N,dt):
        """Init, define number of basis function and time step for integration

        # Arguments
            N: Number of basis functions
            dt: time step

        # Returns


        # Examples
        ```

        ```
        """

        self.a_z=48
        self.a_x=2
        self.N=N
        self.dt=dt

#-----------------------------------------------------------------------------------1
    def precalculate(self,N,dof,dt):



        self.c=np.exp(-self.a_x*np.linspace(0,1,self.N))
        sigma2=np.power((np.diff(self.c)/2),2)
        self.sigma2=np.append(sigma2,sigma2[-1])

        self.tau = np.zeros(dof)
        self.goal = np.zeros(dof)
        self.y0 = np.zeros(dof)
        self.dy0 = np.zeros(dof)
        self.w = np.zeros(self.N,dof)

#-----------------------------------------------------------------------------------2
    def values(self,N,dt,tau,y0,dy0,goal,w):

        dof=len(goal)

        self.N=N
        self.dt=dt

        self.tau = np.zeros(dof)
        self.goal = np.zeros(dof)
        self.y0 = np.zeros(dof)
        self.dy0 = np.zeros(dof)
        self.w = np.zeros((self.N,dof))

        np.copyto(self.tau , tau)
        np.copyto(self.goal , goal)
        np.copyto(self.y0 , y0)
        np.copyto(self.dy0 , dy0 )

        np.copyto(self.w , w )


        self.c=np.exp(-self.a_x*np.linspace(0,1,self.N))
        sigma2=np.power((np.diff(self.c)/2),2)
        self.sigma2=np.append(sigma2,sigma2[-1])



#-----------------------------------------------------------------------------------3
    def track(self,t,y,yd,ydd):
        """Compute wights.

        # Arguments
            N:

        # Returns


        # Examples
        ```

        ```
        """
        dof = len(y[1])
        epsilon = 1.0e-8
        trj_len=len(t)

       	self.t_or = np.zeros((trj_len,dof))
        self.s = np.zeros((trj_len,dof))
        self.v = np.zeros((trj_len,dof))
        self.a = np.zeros((trj_len,dof))
        '''
        np.copyto(self.t_or , t)
        np.copyto(self.s , y)
        np.copyto(self.v , yd)
        np.copyto(self.a , ydd)'''


        self.t_or = np.asarray(t)
        self.s = np.asarray(y)
        self.v = np.asarray(yd)
        self.a = np.asarray(ydd)


        self.tau = np.zeros(dof)
        self.goal = np.zeros(dof)
        self.y0 = np.zeros(dof)
        self.dy0 = np.zeros(dof)
        self.ddy0 = np.zeros(dof)


        np.copyto(self.tau , self.t_or[-1,:])
        np.copyto(self.goal , self.s[-1,:])
        np.copyto(self.y0 , self.s[0,:])
        np.copyto(self.dy0 , self.v[0,:] )
        np.copyto(self.ddy0 , self.a[0,:])


        self.w=np.zeros([self.N,dof])

        self.c=np.exp(-self.a_x*np.linspace(0,1,self.N))
        sigma2=np.power((np.diff(self.c)/2),2)
        self.sigma2=np.append(sigma2,sigma2[-1])

        ft=np.array([np.power(self.tau,2)]*self.t_or.shape[0])*ydd- self.a_z * (self.a_z/4  * (np.array([self.goal]*self.t_or.shape[0]) - y) - np.array([self.tau]*self.t_or.shape[0]) * yd);




        for j in range(0,dof):

            x=np.exp((-self.a_x/self.tau[j])*self.t_or[:,j])

            #A = np.empty

            A=np.zeros((self.N,self.t_or.shape[0]))

            for i in range(0,self.t_or.shape[0]-1):
                psi = np.exp(-0.5*np.power((x[i]-self.c),2)/self.sigma2)
                fac=x[i]/np.sum(psi)
                psi=fac*psi
                psi[np.where(psi<epsilon)] = 0


                A[:,i]=psi


            self.w[:,j] = np.linalg.lstsq(np.transpose(A), ft[:,j])[0]




#-----------------------------------------------------------------------------------4
    def joint(self):
        """Integrate joints.

        # Arguments
            N:

        # Returns


        # Examples
        ```

        ```
        """


        time_steps = int(np.round(np.max(self.tau)/self.dt))

        dt = self.tau/time_steps

        self.t = np.zeros((time_steps,self.w.shape[1]))
        self.Y = np.zeros((time_steps,self.w.shape[1]))
        self.dY = np.zeros((time_steps,self.w.shape[1]))
        self.ddY = np.zeros((time_steps,self.w.shape[1]))

        for j in range(0,self.w.shape[1]):



            y = self.Y[0,j] = self.y0[j]

            z = self.dY[0,j] = self.dy0[j]*self.tau[j]

            x = 1



            for i in range(1,time_steps):


                #state = self.DMP_integrate(state, dt )


                psi = np.exp(-0.5*np.power((x-self.c),2)/self.sigma2)

                fx = np.sum((self.w[:,j]*x)*psi) / np.sum(psi)


                dx = (-self.a_x*x)/self.tau[j]
                dz = self.a_z * (self.a_z/4 * (self.goal[j] - y) - z) + fx
                dy = z


                dz = dz/self.tau[j]
                dy = dy/self.tau[j]

                x = x + dx*dt[j]
                y = y + dy*dt[j]
                z = z + dz*dt[j]


                self.Y[i,j] = y
                self.dY[i,j] = dy
                self.ddY[i-1,j] = dz / self.tau[j]
                self.t[i,j] = self.t[i-1,j] + dt[j]










#-----------------------------------------------------------------------------------5
    def cartesian(self):

        self.joint

        time_steps = int(np.round(np.max(self.tau)/self.dt))

        dt = self.tau/time_steps



#-----------------------------------------------------------------------------------6
    def cart_track(self,t,y,yd,ydd):

        pos = y[:][0:3]
        dpos = yd[:][0:3]
        ddpos = ydd[:][0:3]
        self.track(t,pos,dpos,ddpos)



#-----------------------------------------------------------------------------------7
    def plot_j(self,modul):

        import matplotlib.pyplot as plt

        plt.figure(1,figsize=[10,10])
        plt.subplot(311)
        plt.plot(self.t,self.Y)
        if modul == 1:
            plt.plot(self.t_or,self.s)
        plt.title('Discplacment')

        plt.subplot(312)
        plt.plot(self.t,self.dY)
        if modul == 1:
            plt.plot(self.t_or,self.v)
        plt.title('Velocity')

        plt.subplot(313)
        plt.plot(self.t,self.ddY)
        if modul == 1:
            plt.plot(self.t_or,self.a)
        plt.title('Aceleration')

        plt.show()
#-----------------------------------------------------------------------------------8
    def plot_c(self):

        import matplotlib.pyplot as plt

        plt.figure(1,figsize=[10,10])
        plt.subplot(311)
        plt.plot(self.t,self.Y)
        plt.plot(self.t_or,self.s)
        plt.title('Discplacment')

        plt.subplot(312)
        plt.plot(self.t,self.dY)
        plt.plot(self.t_or,self.v)
        plt.title('Velocity')

        plt.subplot(313)
        plt.plot(self.t,self.ddY)
        plt.plot(self.t_or,self.a)
        plt.title('Aceleration')

        plt.show()

    '''def toROS(self,ROS_DMP):
	ROS_DMP.N = self.N
	ROS_DMP.y0 = self.y0.tolist()
	ROS_DMP.goal = self.goal.tolist()
	#ROS_DMP.dp0 = self.dy0
	ROS_DMP.a_z = self.a_z
	ROS_DMP.b_z = self.a_z*4
	ROS_DMP.a_x = self.a_x
	ROS_DMP.d_t = self.dt

	ROS_DMP.tau = self.tau.tolist()
	ROS_DMP.c = self.c.tolist()
	ROS_DMP.sigma = self.sigma2.tolist()
	ROS_DMP.w = self.w.tolist()
	return ROS_DMP'''

"""

DMP34=DMP(25,0.01)
#DMP34.values(25,0.01,[10,20],[2,4],[1,3],np.zeros((25,2)))
#faza=DMP34.joint()

#
time = np.array((np.linspace(0,10,100),np.linspace(0,5,100))).transpose()
pot = np.array((6+10*0.5*np.power(np.linspace(0,10,100),2),200+3*0.5*np.power(np.linspace(0,5,100),2))).transpose()
hitrost = np.array((10*np.linspace(0,10,100),3*np.linspace(0,5,100))).transpose()
pospesek = np.array((np.linspace(10,10,100),np.linspace(3,3,100))).transpose()

DMP34.track(time,pot,hitrost,pospesek)
#
#DMP34.joint()
DMP11=DMP(25,0.01)
DMP11.values(DMP34.N,DMP34.dt,DMP34.tau,DMP34.y0,DMP34.dy0,DMP34.goal,DMP34.w)

DMP11.joint()
trj3=DMP11.ddY
trj2=DMP11.dY
trj1=DMP11.Y
#np.exp(-0.5*np.power((x-DMP34.c),2)/DMP34.sigma2)

w=DMP34.w
#data=DMP34.data

DMP11.plot_j(0)
#%%
DMP34.tau[0]=4
DMP34.joint()
DMP34.plot_j()
DMP34.tau[1]=1

"""
