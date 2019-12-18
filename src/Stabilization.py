import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from Optimization import real_time_optimize_path
from MeshFlow import motion_propagate
from MeshFlow import mesh_warp_frame
from MeshFlow import generate_vertex_profiles
from scipy.signal import medfilt

class Stabilizer:
    def __init__(self):
        # block of size in mesh
        self.PIXELS = 16
        # buffer size: number frames in the remained lastest past
        self.BUFFER_SZ = 40

        # preserve aspect ratio
        self.HORIZONTAL_BORDER = 30

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=1000,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

        self.frame_n = 0

    def init(self, first_frame):
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        self.VERTICAL_BORDER = (
            self.HORIZONTAL_BORDER*self.old_gray.shape[1])/self.old_gray.shape[0]
        self.VERTICAL_BORDER = int(self.VERTICAL_BORDER)

        self.frame_n = 1

        self.frame_width, self.frame_height = first_frame.shape[1], first_frame.shape[0]

        # x_motion_meshes = []; y_motion_meshes = []
        self.x_paths = np.zeros((int(self.frame_height/self.PIXELS),
                                 int(self.frame_width/self.PIXELS), 1))
        self.y_paths = np.zeros((int(self.frame_height/self.PIXELS),
                                 int(self.frame_width/self.PIXELS), 1))

        self.x_paths_pre = np.empty_like(self.x_paths)
        self.y_paths_pre = np.empty_like(self.y_paths)

        self.sx_paths = np.empty_like(self.x_paths)
        self.sy_paths = np.empty_like(self.y_paths)

        self.x_motion_mesh = np.zeros(
            (int(self.frame_height/self.PIXELS), int(self.frame_width/self.PIXELS)), dtype=float)
        self.y_motion_mesh = np.zeros(
            (int(self.frame_height/self.PIXELS), int(self.frame_width/self.PIXELS)), dtype=float)

        self.good_old = np.zeros((self.frame_height*self.frame_width, 3))
        self.good_old_matrix = np.zeros(
            (self.frame_height, self.frame_width, 3))
    #     good_new_matrix=np.zeros((self.frame_height, self.frame_width, 2))
        index = 0
        for i in range(self.frame_height):
            index = i*self.frame_width
            self.good_old[index:index+self.frame_width,
                          0] = np.arange(self.frame_width)
            self.good_old[index:index+self.frame_width, 1] = i
            self.good_old[index:index+self.frame_width, 2] = 1

        for i in range(self.frame_height):
            self.good_old_matrix[i, :, 0] = np.arange(self.frame_width)
            self.good_old_matrix[i, :, 1] = i
            self.good_old_matrix[i, :, 2] = 1

        # self.temp_good_old = self.good_old.copy()
        self.old_frame = first_frame

        self.retval = cv2.optflow.createOptFlow_DeepFlow()

    def stabilize(self, new_frame: np.ndarray) -> np.ndarray:
        self.frame_n += 1
        frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        flow = np.zeros((self.frame_height, self.frame_width, 2))
        flow = self.retval.calc(self.old_gray, frame_gray, flow)

        good_new = np.copy(self.good_old)
        # good_new_matrix=np.copy(good_old_matrix)

        index = 0
        for i in range(self.frame_height):
            index = i*self.frame_width
            good_new[index:index+self.frame_width, 0] += flow[i, :, 0]
            good_new[index:index+self.frame_width, 1] += flow[i, :, 1]

        x_motion_mesh, y_motion_mesh = motion_propagate(
            self.good_old, good_new, new_frame, self.good_old_matrix)
        self.x_paths, self.y_paths = generate_vertex_profiles(
            self.x_paths, self.y_paths, x_motion_mesh, y_motion_mesh)

        if (self.frame_n > self.BUFFER_SZ):
            self.x_paths = self.x_paths[:, :, 1:]
            self.y_paths = self.y_paths[:, :, 1:]

        if self.frame_n == 2:
            tmp = 1
            while (tmp <= 2):
                self.sx_paths, self.sy_paths, self.x_paths_pre, self.y_paths_pre = self.__stabilize(
                    self.x_paths, self.y_paths, self.x_paths_pre, self.y_paths_pre, self.sx_paths, self.sy_paths, tmp)
                new_x_motion_meshes, new_y_motion_meshes = self.__get_frame_warp(
                    self.x_paths, self.y_paths, self.sx_paths, self.sy_paths)
                if tmp == 1:
                    new_frame = self.__generate_stabilized_video(
                        self.old_frame, tmp, new_x_motion_meshes, new_y_motion_meshes)
                else:
                    new_frame = self.__generate_stabilized_video(
                        new_frame, tmp, new_x_motion_meshes, new_y_motion_meshes)
                tmp += 1
        else:
            self.sx_paths, self.sy_paths, self.x_paths_pre, self.y_paths_pre = self.__stabilize(
                self.x_paths, self.y_paths, self.x_paths_pre, self.y_paths_pre, self.sx_paths, self.sy_paths, self.frame_n)
            new_x_motion_meshes, new_y_motion_meshes = self.__get_frame_warp(
                self.x_paths, self.y_paths, self.sx_paths, self.sy_paths)
            new_frame = self.__generate_stabilized_video(
                new_frame, self.frame_n, new_x_motion_meshes, new_y_motion_meshes)

        self.old_frame = new_frame.copy()
        self.old_gray = frame_gray.copy()
        # self.good_old = temp_good_old.copy()
        return new_frame

    def __get_frame_warp(self, x_paths, y_paths, sx_paths, sy_paths):
        """
        @param: x_motion_meshes is the motion vectors on
                mesh vertices in x-direction
        @param: y_motion_meshes is the motion vectors on
                mesh vertices in y-direction
        @param: x_paths is motion vector accumulation on
                mesh vertices in x-direction
        @param: y_paths is motion vector accumulation on
                mesh vertices in y-direction
        @param: sx_paths is the optimized motion vector
                accumulation in x-direction
        @param: sx_paths is the optimized motion vector
                accumulation in x-direction

        Returns:
                returns a update motion mesh for each frame
                with which that needs to be warped
        """

        # U = P-C
        # x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
        # y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
        new_x_motion_meshes = sx_paths-x_paths
        new_y_motion_meshes = sy_paths-y_paths
        return new_x_motion_meshes, new_y_motion_meshes

    def __stabilize(self, x_paths, y_paths, x_paths_pre, y_paths_pre, sx_paths, sy_paths, frame_n):
        """
        @param: x_paths is motion vector accumulation on
                mesh vertices in x-direction
        @param: y_paths is motion vector accumulation on
                mesh vertices in y-direction

        Returns:
                returns optimized mesh vertex profiles in
                x-direction & y-direction
        """

        # optimize for smooth vertex profiles
        sx_paths, x_paths_pre = real_time_optimize_path(
            x_paths, x_paths_pre, sx_paths, frame_n)
        sy_paths, y_paths_pre = real_time_optimize_path(
            y_paths, y_paths_pre, sy_paths, frame_n)
        return [sx_paths, sy_paths, x_paths_pre, y_paths_pre]

    def __generate_stabilized_video(self, frame, frame_n, new_x_motion_meshes, new_y_motion_meshes):
        """
        @param: cap is the cv2.VideoCapture object that is
                instantiated with given video
        @param: x_motion_meshes is the motion vectors on
                mesh vertices in x-direction
        @param: y_motion_meshes is the motion vectors on
                mesh vertices in y-direction
        @param: new_x_motion_meshes is the updated motion vectors
                on mesh vertices in x-direction to be warped with
        @param: new_y_motion_meshes is the updated motion vectors
                on mesh vertices in y-direction to be warped with
        """

        new_x_motion_mesh = new_x_motion_meshes[:, :, -1]
        new_y_motion_mesh = new_y_motion_meshes[:, :, -1]

        # mesh warping
        new_frame = mesh_warp_frame(
            frame, new_x_motion_mesh, new_y_motion_mesh)
        new_frame = new_frame[self.HORIZONTAL_BORDER:-self.HORIZONTAL_BORDER,
                              self.VERTICAL_BORDER:-self.VERTICAL_BORDER, :]
        new_frame = cv2.resize(
            new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        return new_frame

    def plot_vertex_profiles(self, x_paths, sx_paths):
        """
        @param: x_paths is original mesh vertex profiles
        @param: sx_paths is optimized mesh vertex profiles

        Return:
                saves equally spaced mesh vertex profiles
                in directory '<PWD>/results/'
        """

        # plot some vertex profiles
        for i in range(0, x_paths.shape[0]):
            for j in range(0, x_paths.shape[1], 10):
                plt.plot(x_paths[i, j, :])
                plt.plot(sx_paths[i, j, :])
                plt.savefig('../results/paths/'+str(i)+'_'+str(j)+'.png')
                plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='./0.avi', help='Input video source')
    parser.add_argument('output', type=str, default='./0_stable.avi', help='Output video path')
    args = parser.parse_args()

    stabilizer = Stabilizer()
    
    cap = cv2.VideoCapture(args.input)
    ret, first_frame = cap.read()
    if not ret:
        print('Your input video is unreadable')
        sys.exit(-1)

    stabilizer.init(first_frame)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (stabilizer.frame_width, stabilizer.frame_height))

    while (1):
        ret, new_frame = cap.read()
        if not ret:
            break

        stabilizer.stabilize(new_frame)
        out.write(new_frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print('Done')
