import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Optimization import real_time_optimize_path
from MeshFlow import motion_propagate
from MeshFlow import mesh_warp_frame
from MeshFlow import generate_vertex_profiles
from scipy.signal import medfilt


# block of size in mesh
PIXELS = 16

# motion propogation radius
# RADIUS = 300

#buffer size: number frames in the remained lastest past
BUFFER_SZ = 40

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                    qualityLevel = 0.3,
                    minDistance = 7,
                    blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

def measure_performance(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(method.__name__+' has taken: '+str(end_time-start_time)+' sec')
        return result
    return timed

@measure_performance
def stabilize(x_paths, y_paths, x_paths_pre, y_paths_pre, sx_paths, sy_paths, frame_n):
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
    sx_paths, x_paths_pre = real_time_optimize_path(x_paths, x_paths_pre, sx_paths, frame_n)
    sy_paths, y_paths_pre = real_time_optimize_path(y_paths, y_paths_pre, sy_paths, frame_n)
    return [sx_paths, sy_paths, x_paths_pre, y_paths_pre]


def plot_vertex_profiles(x_paths, sx_paths):
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


def get_frame_warp(x_paths, y_paths, sx_paths, sy_paths):
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


@measure_performance
def generate_stabilized_video(frame, frame_n, new_x_motion_meshes, new_y_motion_meshes):
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
    new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
    new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
    new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    return new_frame

if __name__ == '__main__':
    
    start_time = time.time()
    # get video properties
#     file_name = sys.argv[1]
    file_name='0.avi'
    
    cap = cv2.VideoCapture(file_name)

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # preserve aspect ratio
    
    global HORIZONTAL_BORDER
    HORIZONTAL_BORDER = 30

    global VERTICAL_BORDER
    VERTICAL_BORDER = (HORIZONTAL_BORDER*old_gray.shape[1])/old_gray.shape[0]
    VERTICAL_BORDER=int(VERTICAL_BORDER)

    frame_n=1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # x_motion_meshes = []; y_motion_meshes = []
    x_paths = np.zeros((int(frame_height/PIXELS), int(frame_width/PIXELS), 1))
    y_paths = np.zeros((int(frame_height/PIXELS), int(frame_width/PIXELS), 1))
    
    x_paths_pre=np.empty_like(x_paths)
    y_paths_pre=np.empty_like(y_paths)

    sx_paths=np.empty_like(x_paths)
    sy_paths=np.empty_like(y_paths)

    out = cv2.VideoWriter('0_stable.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    x_motion_mesh = np.zeros((int(frame_height/PIXELS), int(frame_width/PIXELS)), dtype=float)
    y_motion_mesh = np.zeros((int(frame_height/PIXELS), int(frame_width/PIXELS)), dtype=float)

    good_old=np.zeros((frame_height*frame_width, 2))
    good_old_matrix=np.zeros((frame_height, frame_width, 2))
#     good_new_matrix=np.zeros((frame_height, frame_width, 2))
    index=0
    for i in range(frame_height):
            index=i*frame_width
            good_old[index:index+frame_width,0]=np.arange(frame_width)
            good_old[index:index+frame_width,1]=i
    
    for i in range(frame_height):
        good_old_matrix[i,:,0]=np.arange(frame_width)
        good_old_matrix[i,:,1]=i

    temp_good_old=good_old.copy()

#     fast = cv2.FastFeatureDetector_create(110)
    # Initiate ORB detector
    # orb = cv2.ORB_create(130)

    retval=cv2.optflow.createOptFlow_DeepFlow()
    while (1):
        # # find corners in it
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # # find and draw the keypoints
        # kp = fast.detect(old_frame,None)

        # find the keypoints with ORB
        # kp = orb.detect(old_frame,None)
        # compute the descriptors with ORB
        # kp, des = orb.compute(old_frame, kp)

        # p0=np.asarray([[p.pt[0],p.pt[1]] for p in kp])
        # p0=np.expand_dims(p0, axis=1)
        # p0=p0.astype(np.float32)

        ret, frame = cap.read()
        if (not(ret)): break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_n+=1

        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # # Select good points
        # good_new = p1[st==1]
        # good_old = p0[st==1]

        # estimate motion mesh for old_frame
        # take a long time

        start_time = time.time()
        
        
        # flow = cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow=cv2.optflow.calcOpticalFlowSF(old_frame,frame,3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10)
        
        flow=np.zeros((frame_height, frame_width, 2))
        flow=retval.calc(old_gray, frame_gray, flow)
        
        good_new=np.copy(good_old)
        # good_new_matrix=np.copy(good_old_matrix)

        index=0
        for i in range(frame_height):
                index=i*frame_width
                good_new[index:index+frame_width,0]+=flow[i,:,0]
                good_new[index:index+frame_width,1]+=flow[i,:,1]
        
        # for i in range(frame_height):
        #         good_new_matrix[i,:,0]+=flow[i,:,0]
        #         good_new_matrix[i,:,1]+=flow[i,:,1]

        # for i in range(frame_height/PIXELS):
        #         for j in range(frame_width/PIXELS):
        #                 x_motion_mesh[i][j]=(np.median(flow[i*PIXELS:(i+1)*PIXELS,j*PIXELS:(j+1)*PIXELS,0]))
        #                 y_motion_mesh[i][j]=(np.median(flow[i*PIXELS:(i+1)*PIXELS,j*PIXELS:(j+1)*PIXELS,1]))

        # x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
        # y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])

        x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame,good_old_matrix)
        # x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
        end_time = time.time()
        print('motion_propagate'+' has taken: '+str(end_time-start_time)+' sec')

        # try:
        #     x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
        #     y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
        # except:
        #     x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
        #     y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)
        
        # if frame_n > BUFFER_SZ:
        #     x_motion_meshes=x_motion_meshes[:,:,(frame_n-BUFFER_SZ):]
        #     y_motion_meshes=y_motion_meshes[:,:,(frame_n-BUFFER_SZ):]

        # generate vertex profiles
        x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        if (frame_n > BUFFER_SZ):
            x_paths=x_paths[:,:,1:]
            y_paths=y_paths[:,:,1:]

        # propogate motion vectors and generate vertex profiles
        # x_motion_meshes, y_motion_meshes, x_paths, y_paths = read_video(frame_gray,x_paths,y_paths,x_motion_meshes,y_motion_meshes)
        
        # stabilize the vertex profiles
        if frame_n==2:
            tmp=1
            while (tmp <= 2):
                sx_paths, sy_paths, x_paths_pre, y_paths_pre = stabilize(x_paths, y_paths, x_paths_pre, y_paths_pre, sx_paths, sy_paths, tmp)
                new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_paths, y_paths, sx_paths, sy_paths)
                if tmp==1:
                    new_frame=generate_stabilized_video(old_frame, tmp, new_x_motion_meshes, new_y_motion_meshes)
                else:
                    new_frame=generate_stabilized_video(frame, tmp, new_x_motion_meshes, new_y_motion_meshes)
                tmp+=1
        else:
            sx_paths, sy_paths, x_paths_pre, y_paths_pre = stabilize(x_paths, y_paths, x_paths_pre, y_paths_pre, sx_paths, sy_paths, frame_n)           
            new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_paths, y_paths, sx_paths, sy_paths)
            new_frame=generate_stabilized_video(frame, frame_n, new_x_motion_meshes, new_y_motion_meshes)
        
        out.write(new_frame)

        cv2.imshow('new_frame',new_frame)
        if (cv2.waitKey(1) & 0xFF==ord('q')):
            break 

        old_frame = frame.copy()
        old_gray = frame_gray.copy()

        good_old=temp_good_old.copy()

        print('frame ', frame_n, ': -----------------------------')

#     visualize optimized paths
    plot_vertex_profiles(x_paths, sx_paths)

    cap.release()
    cv2.destroyAllWindows()
        # visualize optimized paths
    #     plot_vertex_profiles(x_paths, sx_paths)

        # get updated mesh warps
        # x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths)

        # apply updated mesh warps & save the result
        # generate_stabilized_video(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes)
        # print('Time elapsed: ', str(time.time()-start_time))