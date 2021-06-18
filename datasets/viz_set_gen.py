import argparse
import os
import shutil


def extract_samples_from_sets(sample_dir, viz_sample_dir, train_sets):

    # Viz samples need to be stored in a subdirectory (dir/0/)
    if os.path.isdir(viz_sample_dir) == False:
        os.mkdir(viz_sample_dir)
    viz_sample_subdir = os.path.join(viz_sample_dir, "0/")
    if os.path.isdir(viz_sample_subdir) == False:
        os.mkdir(viz_sample_subdir)

    train_sample_idx = 0

    for train_set in train_sets:

        print(f"Processing training set: '{train_set}'")

        train_set_path = os.path.join(sample_dir, train_set)

        if os.path.isdir(train_set_path) == False:
            raise Exception(
                f"Given 'train_set_path' is not a directory ({train_set_path})")

        for set_sample_idx in range(len(os.listdir(train_set_path))):

            print(f"   eval sample {train_sample_idx} ( <- {set_sample_idx} )")

            file_path = os.path.join(train_set_path, f"{set_sample_idx}.gz")
            copy_path = os.path.join(
                viz_sample_subdir, f"{train_sample_idx}.gz")

            try:
                shutil.copy(file_path, copy_path)
            except:
                s = f"Could not copy file\n    source: {file_path}\n    " \
                    f"destination: {copy_path}"
                raise Exception(s)

            train_sample_idx += 1

            break

    print(f"Finished copying {train_sample_idx} evaluation samples")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-dir", type=str, default="data_gen/artificial/",
        help="Path to raw samples root directory"
    )
    parser.add_argument(
        "--viz-sample-dir", type=str, default="datasets/viz_samples/",
        help="Path to viz sample output directory"
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    viz_sample_dir = args.viz_sample_dir

    all_scenes_sets = ["samples_intersection_1",
                       "samples_intersection_2",
                       "samples_intersection_3",
                       "samples_intersection_4",
                       "samples_intersection_5",
                       "samples_intersection_6",
                       "samples_intersection_7",
                       "samples_straight_1",
                       "samples_straight_2",
                       "samples_straight_3",
                       "samples_turn_1",
                       "samples_triangle_intersection_1",
                       "samples_triangle_intersection_2",
                       "samples_triangle_intersection_3",
                       "samples_triangle_intersection_4",
                       "samples_triangle_intersection_5",
                       "samples_triangle_intersection_6",
                       "samples_triangle_intersection_7",
                       "samples_triangle_intersection_8",
                       "samples_y_intersection_1",
                       "samples_y_intersection_2",
                       "samples_y_intersection_3",
                       "samples_y_intersection_4",
                       "samples_roundabout_1",
                       "samples_roundabout_2"]

    extract_samples_from_sets(
        sample_dir, viz_sample_dir, all_scenes_sets)

