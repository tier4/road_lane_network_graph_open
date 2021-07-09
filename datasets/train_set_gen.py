import argparse
import os
import shutil


def extract_samples_from_sets(dataset_dir_path, train_dir, train_sets):

    train_sample_idx = 0

    # TODO: Make samples dynamically arranged into subdirs
    traing_dir_subdir = os.path.join(train_dir, '0/')
    if os.path.isdir(traing_dir_subdir) == False:
        os.mkdir(traing_dir_subdir)

    for train_set in train_sets:

        print(f"Processing training set: '{train_set}'")

        train_set_path = os.path.join(dataset_dir_path, train_set)

        if os.path.isdir(train_set_path) == False:
            raise Exception(
                f"Given 'train_set_path' is not a directory ({train_set_path})")

        for set_sample_idx in range(len(os.listdir(train_set_path))):

            print(f"   training sample {train_sample_idx} (<-{set_sample_idx})")

            file_path = os.path.join(train_set_path, f"{set_sample_idx}.gz")
            copy_path = os.path.join(
                traing_dir_subdir, f"{train_sample_idx}.gz")

            try:
                shutil.copy(file_path, copy_path)
            except:
                s = "Could not copy file" \
                     f"\n    source: {file_path}" \
                     f"\n    destination: {copy_path}"
                raise Exception(s)

            train_sample_idx += 1

    print(f"Finished copying {train_sample_idx} training samples")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-dir", type=str, default="data_gen/artificial/",
        help="Path to raw samples root directory"
    )
    parser.add_argument(
        "--train-dir", type=str, default="datasets/training_samples/",
        help="Path to training sample output directory"
    )
    args = parser.parse_args()
    
    dataset_dir_path = args.sample_dir

    train_dir = args.train_dir
    if os.path.isdir(train_dir) == False:
        os.mkdir(train_dir)

    base_train_set = ["samples_intersection_1",
                      "samples_intersection_2",
                      "samples_intersection_4",
                      "samples_intersection_7",
                      "samples_straight_1",
                      "samples_straight_2",
                      "samples_straight_3",
                      "samples_turn_1",
                      "samples_triangle_intersection_1",
                      "samples_triangle_intersection_3",
                      "samples_triangle_intersection_4",
                      "samples_triangle_intersection_5",
                      "samples_triangle_intersection_6",
                      "samples_y_intersection_1",
                      "samples_roundabout_1",
                      "samples_lane_fork",
                      "samples_lane_merge"]

    extract_samples_from_sets(dataset_dir_path, train_dir, base_train_set)
