import numpy as np
import json
import os

prefix = 'image_'
suffix = '.json'

class TrajectoryLoader:
    """
    Static helper class for loading trejectoires from json files
    """
    def load(file):
        """
        Loads trajectory from file

        load(file) ->  trajectory containing all points in a form 'point = [x,y,t]'
        file -> file containing trajectory in json format
        """
        try:
            json_data = open(file).read()

            data = json.loads(json_data)
            path = data['Path']
            path = path.split('], [')
            path[0] = path[0][1:]
            path[-1] = path[-1][:-1]

            points = []

            for point in path:
                point = point.split(',')
                point = [float(x) for x in point]
                if len(point) != 3:
                    raise Exception('Error in file ' + file)
                points.append(point)
            points = np.array(points)
            if np.where(points[:,2] == 0)[0].size > 1:
                print('File ' + file + 'is corrupted')
            return points
        except:
            print('Could not load file ' + file)


    def getAvaliableTrajectoriesNumbers(folder):
        """
        Checks folder for all json files containing trajectories and returns its numbers gained from filenames
        Each number n coresponds to the n-th image from the MNIST dataset

        getAvaliableTrajectoriesNumbers(folder) -> sorted list of numbers of avaliable trajectories inf the folder
        folder -> the string path of the folder to check
        """
        datoteke =  [f for f in os.listdir(folder)]
        avaliable = []
        for datoteka in datoteke:
            if datoteka.startswith(prefix) and datoteka.endswith(suffix):
                number = datoteka[len(prefix):]
                number = int(number[:-len(suffix)])
                avaliable.append(number)
        return sorted(avaliable)

    def getTrajectoryFile(folder, n):
        """
        Returns the string path to the file with the n-th trajectory in the given folder

        getTrajectoryFile(folder, n) -> string path to the trajectory
        folder -> the string path to the folder containing the trajectory files
        n -> the sequential number of the desired trajectory
        """
        datoteke =  [f for f in os.listdir(folder)]
        name = prefix + str(n) + suffix
        if name in datoteke:
            return folder + "/" + name
        return False

    def loadNTrajectory(folder,n):
        """
        Loads n-th trajectory from the given folder

        loadNTrajectory(folder,n) ->
        folder -> the string path to the folder containing the trajectory files
        n -> the sequential number of the desired trajectory
        """
        return trajectory_loader.load(trajectory_loader.getTrajectoryFile(folder,n))
