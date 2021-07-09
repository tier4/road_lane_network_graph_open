import argparse
import cv2
import os
import pickle
import gzip

'''
HOW TO USE

1. Check folder structure and context images:
    data: Contains context images used for generating samples.
    samples: Pickled sample files will be generated here.

2.  Run program
     python artificial_sample_gen.py data/ samples/

'''

# Global variables
ij_waypoints = []
ij_maneuverpoints = []


def recordMouseCoordinates(event, i, j, flags, param):
    # Grab references to the global variable
    global ij_waypoints
    global ij_maneuverpoints

    # Catch mouse clicks and store their location
    if event == cv2.EVENT_LBUTTONDOWN:
        ij_waypoints.append((i, j))
        # Visualize waypoint path
        if len(ij_waypoints) >= 2:
            cv2.line(img_clone, ij_waypoints[-2],
                        ij_waypoints[-1], (0, 0, 255), 2)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        ij_maneuverpoints.append((i, j))
        # Visualize maneuver points
        cv2.circle(img_clone, ij_maneuverpoints[-1], 4, (255, 0, 0), 2)


def write_compressed_pickle(obj, filename, write_dir):
    '''Converts an object into byte representation and writes a compressed file.

    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    path = os.path.join(write_dir, f"{filename}.gz")
    pkl_obj = pickle.dumps(obj)
    try:
        with gzip.open(path, "wb") as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)


def read_compressed_pickle(path):
    '''Reads a compressed binary file and return the object.

    Args:
        path (str): Path to the file (incl. filename)
    '''
    try:
        with gzip.open(path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def gen_sample(sample_idx, context_tensor, waypoints, maneuverpoints, output_dir):
    '''Package a set of data objects as a dictionary representing one sampel.

    Args:
        sample_idx (int): For use in sample filename.
        context_tensor (np array): Tensor a dense representing of the
                                   environment. Can be grayscale or RGB (i.e. 
                                   of size (n,n) or (n,n,3))
        waypoints (list): List of float tuples (i, j).
        maneuverpoints (list): List of float tuples (i, j).
        output_dir (str): Path to output directory.
    '''
    # Remove duplicate entries from lists
    waypoints = unique(waypoints)
    maneuverpoints = unique(maneuverpoints)
    # Convert context_tensor to grayscale
    if len(context_tensor.shape) == 3:
        context_tensor = cv2.cvtColor(context_tensor, cv2.COLOR_BGR2GRAY)
    # Fill dictionary
    sample_dic = {}
    sample_dic["context"] = context_tensor
    sample_dic["traj"] = waypoints
    sample_dic["maneuver"] = maneuverpoints

    filename = f"{sample_idx}"
    write_compressed_pickle(sample_dic, filename, output_dir)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "output_dir", type=str, help="Path to the directory to generate samples")
parser.add_argument(
    "--scene-dir", type=str, default="data_gen/artificial/scenes/",
    help="Path to the directory containing images"
),
args = parser.parse_args()
data_path = args.scene_dir
sample_path = args.output_dir

# IMAGE COORDINATE FRAME (i,j)
#   _______i
#  |0,0
#  |
#  |
#  j     i,j

# Generate a list of images from directory
s = "Generate artificial samples from input images\n    " \
    "Press 'n' to complete one sample, and 'q' to go to next image\n"
print(s)
sample_idx = 0
image_names = os.listdir(data_path)
for image_name in image_names:

    next_image_flag = False
    sample_finished_flag = False

    while next_image_flag == False:

        print(f"sample {sample_idx} ({image_name})")
        next_image_flag = False

        # Read image
        image_path = os.path.join(data_path, image_name)
        context_img = cv2.imread(image_path)

        while sample_finished_flag == False:

            # Create clone image for visualization
            img_clone = context_img.copy()

            # Initialize empty list of (i, j) points
            ij_waypoints = []
            ij_maneuverpoints = []

            # Create window
            window_name = "Context image window"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, recordMouseCoordinates)

            cancel_flag = False
            while True:
                # display the image and wait for a keypress
                cv2.imshow(window_name, img_clone)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    cancel_flag = True
                    break
                elif key == ord("n"):
                    sample_finished_flag = True
                    break
                elif key == ord("q"):
                    sample_finished_flag = True
                    next_image_flag = True
                    break

            # Combine 'context image' and 'coordinate list' into data sample
            if len(ij_waypoints) >= 3 and cancel_flag == False:

                ###################
                # GENERATE SAMPLE
                ###################
                print(ij_waypoints)
                gen_sample(sample_idx, context_img, ij_waypoints,
                           ij_maneuverpoints, sample_path)
                sample_idx += 1

            else:
                print("Skipping sample")

            cv2.destroyAllWindows()

        sample_finished_flag = False


print("\nFinished generating " + str(sample_idx) + " data samples")
