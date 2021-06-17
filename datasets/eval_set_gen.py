import argparse
import os
import shutil


def extract_samples_from_sets(
        test_sample_dir, eval_dir_path, num_samples, test_sets):

    if os.path.isdir(eval_dir_path) == False:
        raise Exception(
            f"Given 'eval_dir_path' is not a directory ({eval_dir_path})")

    eval_sample_idx = 0

    for test_set in test_sets:

        print(f"Processing test set: '{test_set}'")

        test_set_path = os.path.join(test_sample_dir, test_set)

        if os.path.isdir(test_set_path) == False:
            raise Exception(
                f"Given 'test_set_path' is not a directory ({test_set_path})")

        for test_sample_idx in range(num_samples):

            print(f"   eval sample {eval_sample_idx} ( <- {test_sample_idx} )")

            file_path = os.path.join(test_set_path, f"{test_sample_idx}.gz")
            copy_path = os.path.join(eval_dir_path, f"{eval_sample_idx}.gz")

            try:
                shutil.copy(file_path, copy_path)
            except:
                s = f"Could not copy file\n    source: {file_path}\n    " \
                    f"destination: {copy_path}"
                raise Exception(s)

            eval_sample_idx += 1

    print(f"Finished copying {eval_sample_idx+1} evaluation samples")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-sample-dir", type=str, default="datasets/test_samples/",
        help="Path to test sample output directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of test samples to generate for each raw sample"
    )
    parser.add_argument(
        "--training-eval-dir", type=str, default="datasets/training_eval/",
        help="Path to training evaluation sample output directory"
    )
    parser.add_argument(
        "--test-eval-dir", type=str, default="datasets/test_eval/",
        help="Path to test evaluation sample output directory"
    )
    args = parser.parse_args()

    test_sample_dir = args.test_sample_dir
    num_samples = args.num_samples
    training_eval_dir = args.training_eval_dir
    test_eval_dir = args.test_eval_dir

    training_eval_set = ["samples_intersection_1",
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

    test_eval_set = ["samples_triangle_intersection_2",
                     "samples_triangle_intersection_7",
                     "samples_triangle_intersection_8",
                     "samples_y_intersection_2",
                     "samples_y_intersection_3",
                     "samples_y_intersection_4",
                     "samples_roundabout_2",
                     "samples_intersection_3",
                     "samples_intersection_5",
                     "samples_intersection_6",
                     "samples_intersection_8"]

    if os.path.isdir(training_eval_dir) == False:
        os.mkdir(training_eval_dir)
    extract_samples_from_sets(
        test_sample_dir, training_eval_dir, num_samples, training_eval_set)
    
    if os.path.isdir(test_eval_dir) == False:
        os.mkdir(test_eval_dir)
    extract_samples_from_sets(
        test_sample_dir, test_eval_dir, num_samples, test_eval_set)
