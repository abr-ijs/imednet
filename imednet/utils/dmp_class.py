"""
DMP integration in Cartesian or joint coordinates.
"""
import numpy as np

class DMP(object):
    def __init__(self, N, dt):
        """Init, define number of basis function and time step for integration

        # Arguments
            N: Number of basis functions
            dt: time step

        # Returns


        # Examples
        ```

        ```
        """
        self.a_z = 48
        self.a_x = 2
        self.N = N
        self.dt = dt

    def precalculate(self, N, dof, dt):
        self.c = np.exp(-self.a_x * np.linspace(0, 1, self.N))
        sigma2 = np.power((np.diff(self.c)/2), 2)
        self.sigma2 = np.append(sigma2, sigma2[-1])

        self.tau = np.zeros(dof)
        self.goal = np.zeros(dof)
        self.y0 = np.zeros(dof)
        self.dy0 = np.zeros(dof)
        self.w = np.zeros(self.N, dof)

    def values(self, N, dt, tau, y0, dy0, goal, w):

        dof = len(goal)

        self.N = N
        self.dt = dt

        self.tau = np.zeros(dof)
        self.goal = np.zeros(dof)
        self.y0 = np.zeros(dof)
        self.dy0 = np.zeros(dof)
        self.w = np.zeros((self.N, dof))

        np.copyto(self.tau, tau)
        np.copyto(self.goal, goal)
        np.copyto(self.y0, y0)
        np.copyto(self.dy0, dy0)

        np.copyto(self.w, w)

        self.c = np.exp(-self.a_x*np.linspace(0, 1, self.N))
        sigma2 = np.power((np.diff(self.c)/2), 2)
        self.sigma2 = np.append(sigma2, sigma2[-1])


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
        trj_len = len(t)

        self.t_or = np.zeros((trj_len, dof))
        self.s = np.zeros((trj_len, dof))
        self.v = np.zeros((trj_len, dof))
        self.a = np.zeros((trj_len, dof))
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
        np.copyto(self.dy0, self.v[0,:])
        np.copyto(self.ddy0, self.a[0,:])


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

    def joint(self):
        """Integrate joints.

        # Arguments
            N:

        # Returns


        # Examples
        ```

        ```
        """
        time_steps = int(np.round(np.max(self.tau)/self.dt))+1

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

    def cartesian(self):
        self.joint
        time_steps = int(np.round(np.max(self.tau)/self.dt))
        dt = self.tau/time_steps

    def cart_track(self,t,y,yd,ydd):
        pos = y[:][0:3]
        dpos = yd[:][0:3]
        ddpos = ydd[:][0:3]
        self.track(t,pos,dpos,ddpos)

    def plot_j(self, modul = False):

        import matplotlib.pyplot as plt

        plt.figure(1,figsize=[10,10])
        plt.subplot(311)
        plt.plot(self.t,self.Y)
        if modul:
            plt.plot(self.t_or,self.s)
        plt.title('Discplacment')

        plt.subplot(312)
        plt.plot(self.t,self.dY)
        if modul:
            plt.plot(self.t_or,self.v)
        plt.title('Velocity')

        plt.subplot(313)
        plt.plot(self.t,self.ddY)
        if modul:
            plt.plot(self.t_or,self.a)
        plt.title('Aceleration')

        plt.show()

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
