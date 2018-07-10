


# Quaternion operations
from tf.transformations import *


# Numpy for maths
import numpy as np


class JointSpaceDMP(object):
    N = 0
    y0 = []
    goal = []
    a_z = 0
    b_z = 0
    a_x = 0
    d_t = 0.1
    id = 2
    tau = 3
    c = []
    sigma = []
    w = []


class Quarternion(object):
    x = 0
    y = 0
    z = 0
    w = 0


class Point(object):
    x = 0
    y = 0
    z = 0


class Pose(object):
    position = Point()
    orentation = Quarternion()


class CartesianVelocity(object):
    linear = []
    angular = []


class CartesianSpaceDMP(object):
    N = 0
    p0 = Pose()
    goal = Pose()
    dp0 = CartesianVelocity()
    a_z = 0
    b_z = 0
    a_x = 0
    d_t = 0.1
    id = 2
    tau = 3
    c = []
    sigma = []
    w = []


class DMP(object):
    # Declaration of variables to be used inside the object / class
    jointDMP = None
    cartDMP = None
    trainingData = None
    trainingPosition = None
    trainingVelocity = None
    trainingOrientation = None
    trainingTime = None
    numDOF = None
    a_x = None
    a_z = None
    b_z = None
    N = None
    tau = None
    numSamples = None
    c = None
    sigma = None
    wP = None
    wQ = None

    def __init__(self):

        # Prepare the DMP objects that will later be used
        self.jointDMP = JointSpaceDMP()
        self.cartDMP = CartesianSpaceDMP()

        # Prepare the variables that will later be used for DMP calculation
        self.trainingData = []
        self.trainingPosition = []
        self.trainingVelocity = []
        self.trainingOrientation = []
        self.trainingTime = []
        self.numDOF = []

    def __parse_joint_recording(self):
        # Make sure the arrays are clear before proceeding
        self.trainingPosition = []
        self.trainingVelocity = []

        # Extract the initial and final position properties
        self.jointDMP.y0 = self.trainingData[0]
        self.jointDMP.goal = self.trainingData[-1]

        # Get the number of degrees of freedom
        self.numDOF = len(self.trainingData[0].position)

        # Exctract the position and velocity vector
        for sample in self.trainingData:
            self.trainingPosition.append(sample.position)
            self.trainingVelocity.append(sample.velocity)

    def __parse_cartesian_recording(self):
        # Make sure the arrays are clear before proceeding
        self.trainingPosition = []
        self.trainingVelocity = []
        self.trainingOrientation = []

        # For the Cartesian space DMP the number of degrees of freedom CAN be hardcoded as it cannot be anyhow different
        self.numDOF = 3 # For the position part

        # # Exctract the position and velocity vector
        for sample in self.trainingData:
            self.trainingPosition.append([sample.pose.position.x, sample.pose.position.y, sample.pose.position.z])
            self.trainingOrientation.append([sample.pose.orientation.x, sample.pose.orientation.y, sample.pose.orientation.z, sample.pose.orientation.w])

        # Fix the quaternions' signs in cases where they suddenly change (q = -q)
        signChangeIdx = np.where((np.linalg.norm(np.diff(self.trainingOrientation,axis=0),axis=1)) > 0.5)[0]

        # In case the trajectory ends with the sign flipped, add the last index to the indeces array
        if len(signChangeIdx) % 2 != 0:
            signChangeIdx = np.append(signChangeIdx, len(self.trainingOrientation)-1)

        # Change signs where neccesarry
        for i in range(len(signChangeIdx))[::2]:
            self.trainingOrientation[signChangeIdx[i]+1:signChangeIdx[i+1]+1] = np.negative(self.trainingOrientation[signChangeIdx[i]+1:signChangeIdx[i+1]+1])

        # Make sure everyhing is in the right format
        self.trainingPosition = np.asarray(self.trainingPosition)
        self.trainingOrientation = np.asarray(self.trainingOrientation)

        # Estimate the end effector translation velocity
        self.trainingVelocity = np.empty((0, self.numSamples), dtype=np.float32)
        for i in range(self.numDOF):
            self.trainingVelocity = np.append(self.trainingVelocity, [np.divide(np.gradient(np.asarray(self.trainingPosition[:,i])),np.gradient(self.trainingTime))], axis=0)

        self.trainingVelocity = self.trainingVelocity.transpose()

        # Extract the initial and final position properties
        self.cartDMP.p0 = self.trainingData[0].pose
        self.cartDMP.goal = self.trainingData[-1].pose

        # This step is neccesary so we get the orientation with the corrected sign!
        self.cartDMP.goal.orientation = Quaternion(x=self.trainingOrientation[-1][0],
                                                   y=self.trainingOrientation[-1][1],
                                                   z=self.trainingOrientation[-1][2],
                                                   w=self.trainingOrientation[-1][3])

    def __train_position_dmp(self):

        # Don't just copy the data, make sure it is of the right type
        y = np.asarray(self.trainingPosition)
        dy = np.asarray(self.trainingVelocity)

        # Estimate the accelerations
        ddy = np.zeros(dy.shape)
        for i in range(self.numDOF):
            ddy[:,i] = np.divide(np.gradient(dy[:,i]),np.gradient(self.trainingTime))

        # Prepare empty matrices
        ft = np.zeros((self.numSamples, self.numDOF),dtype=np.float32)
        A = np.zeros((self.numSamples, self.N), dtype=np.float32)
        x = np.exp(-self.a_x * self.trainingTime / self.tau)

        # Estimate the forcing term
        for dof in range(self.numDOF):
            ft[:,dof] = ddy[:,dof]*np.square(self.tau) - \
                        self.a_z * (self.b_z * (y[-1][dof] - y[:,dof]) - dy[:,dof] * self.tau)


        for i in range(self.numSamples):
            psi = np.exp(np.divide(-0.5 * np.square(x[i]- self.c), self.sigma))
            A[i,:] = x[i] * np.divide(psi, np.sum(psi))

        # Do linear regression in the least square sense
        w = np.linalg.lstsq(A,ft)[0]

        self.wP = []
        for i in range(self.numDOF):
            #self.wP.append(robot_module_msgs.msg.Float32Array(w[:,i].tolist()))                         #!!!!!!!!!!!!!!!!!
            self.wP.append(w[:, i].tolist())
    def __train_quaternion_dmp(self):

        # Don't just copy the data, make sure it is of the right type
        q = np.asarray(self.trainingOrientation)

        # Calculate a quaternion derivative that is needed to calculate the rotation velocity
        dq = np.zeros(q.shape, dtype=np.float32)
        for i in range(4):
            dq[:,i] = np.divide(np.gradient(q[:,i]),np.gradient(self.trainingTime))

        # Calculate the rotation velocity
        omega = np.empty((0, 3), dtype=np.float32)
        for i in range(self.numSamples):
            omega = np.append(
                omega,
                [2*quaternion_multiply(dq[i,:], quaternion_conjugate(q[i,:]))[0:3]],
                axis=0)

        # Calculate the rotation acceleration
        domega = np.empty(omega.shape, dtype=np.float32)
        for i in range(3):
            domega[:, i] = np.divide(np.gradient(omega[:, i]), np.gradient(self.trainingTime))

        # Prepare empty matrices
        ft = np.zeros((self.numSamples, 3), dtype=np.float32)
        A = np.zeros((self.numSamples, self.N), dtype=np.float32)
        x = np.exp(-self.a_x * self.trainingTime / self.tau)

        # Estimate the forcing term
        for i in range(self.numSamples):
            ft[i, :] = np.square(self.tau) * domega[i, :] \
                        + self.a_z * self.tau * omega[i, :] \
                        - self.a_z * self.b_z * 2 * self.__quat_diff(q[-1, :], q[i, :])


            psi = np.exp(np.divide(-0.5 * np.square(x[i] - self.c), self.sigma))
            A[i, :] = x[i] * np.divide(psi, np.sum(psi))

        # Do linear regression in the least square sense
        w = np.linalg.lstsq(A, ft)[0]

        self.wQ = []
        for i in range(self.numDOF):
            # self.wP.append(robot_module_msgs.msg.Float32Array(w[:,i].tolist()))                         #!!!!!!!!!!!!!!!!!
            self.wP.append(w[:, i].tolist())

    def __quat_diff(self, q1, q2):
        q = quaternion_multiply(q1, quaternion_conjugate(q2))

        log_q = np.array([0,0,0])
        if (np.linalg.norm(q[0:3]) > 1.0e-12):
            log_q = np.arccos(q[3]) * q[0:3] / np.linalg.norm(q[0:3])
            if np.linalg.norm(log_q) > np.pi:
                log_q = (2 * np.pi - 2 * np.arccos(q[3])) * (-q[0:3]) / np.linalg.norm(q[0:3])

        return log_q




    def __train_cart_dmp(self):
        self.__train_position_dmp()
        self.__train_quaternion_dmp()


        pass

    def __clear_all(self):
        self.jointDMP = None
        self.cartDMP = None
        self.trainingData = None
        self.trainingPosition = None
        self.trainingVelocity = None
        self.trainingOrientation = None
        self.trainingTime = None
        self.numDOF = None
        self.a_x = None
        self.a_z = None
        self.b_z = None
        self.N = None
        self.tau = None
        self.numSamples = None
        self.c = None
        self.sigma = None
        self.wP = None
        self.wQ = None
        pass

    def TrainDMP(self, trainingData, id = 1337, numw = 25, a_z = 48.0, a_x = 2.0):

        # Fill in the DMP parameters
        self.N = self.cartDMP.N = self.jointDMP.N = numw
        self.a_z = self.cartDMP.a_z = self.jointDMP.a_z = a_z
        self.b_z = self.cartDMP.b_z = self.jointDMP.b_z = self.jointDMP.a_z / 4
        self.a_x = self.cartDMP.a_x = self.jointDMP.a_x = a_x
        self.trainingData = trainingData

        # Initialize the Gaussian kernel functions
        self.c = np.exp(-self.jointDMP.a_x * np.linspace(0, 1, self.jointDMP.N))
        self.sigma = np.square((np.diff(self.c)*0.75))
        self.sigma = np.append(self.sigma, self.sigma[-1])

        # Compile the time vector
        self.trainingTime = []
        beginTime = (self.trainingData[0].header.stamp.secs * 1.0 + 1e-9 * self.trainingData[0].header.stamp.nsecs)
        for sample in self.trainingData:
            self.trainingTime = np.append(self.trainingTime, sample.header.stamp.secs *1.0 + 1e-9*sample.header.stamp.nsecs - beginTime)
        self.d_t = np.mean(np.diff(self.trainingTime))

        # Tau equals to the duration of the trajectory
        self.tau = self.trainingTime[-1]

        # Get  the number of samples
        self.numSamples = len(self.trainingTime)

        # Extract the message type so we know if we are dealing with a joinspace or Cartesian space DMP
        trainingDataMessageType = str(type(trainingData[0]))

        if (trainingDataMessageType.find('JointState') != -1):
            print('Training a joint space DMP ...')
            self.__parse_joint_recording()
            self.__train_position_dmp()

            # This variables were used with np so they were np.arrays. In order to use them later on we need them as a list
            self.jointDMP.sigma = self.sigma.tolist()
            self.jointDMP.c = self.c.tolist()
            self.jointDMP.tau = self.tau
            self.jointDMP.d_t = self.d_t
            self.jointDMP.id = id

            # Copy the weights
            self.jointDMP.w = self.wP

            # print self.jointDMP.w

            return self.jointDMP

        if (trainingDataMessageType.find('PoseStamped') != -1):
            print('Training a Cartesian space DMP ...')
            self.__parse_cartesian_recording()
            self.__train_cart_dmp()

            # # Copy the weights (I know this looks ridiculous, but I do not know of a better way ... yet!)
            self.cartDMP.w = [self.wP[0],
                              self.wP[1],
                              self.wP[2],
                              self.wQ[0],
                              self.wQ[1],
                              self.wQ[2]]

            # print self.cartDMP.w

            # This variables were used with np so they ere np.arrays. In order to use them later on we need them as a list
            self.cartDMP.sigma = self.sigma.tolist()
            self.cartDMP.c = self.c.tolist()
            self.cartDMP.tau = self.tau
            self.cartDMP.d_t = self.d_t
            self.cartDMP.id = id

            return self.cartDMP



