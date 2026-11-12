# Find all TSV files with the name "results_all.tsv" in subdirectories of the specified main directory.
# Move and rename them: move to specified target directory; rename them results_all_<super>.tsv
# where <super> is the name of the subdirectory containing the original "results_all.tsv" file.

import os
import shutil
import argparse
from rich import print

def collate_results(main_directory: str, target_directory: str) -> None:

    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Walk through all subdirectories of the main directory
    n_moved = 0
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file == "results_all.tsv":
                subdirectory_name = os.path.basename(root)
                new_file_name = f"results_all_{subdirectory_name}.tsv"
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_directory, new_file_name)
                shutil.copy(source_path, target_path)
                n_moved += 1
    print(f"[green]Moved {n_moved} files to [spring_green2]{target_directory}[/spring_green2][/green]")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collate results TSV files")
    parser.add_argument("-m", "--main_directory", type=str, required=True, help="Main directory containing subdirectories with results_all.tsv")
    parser.add_argument("-t", "--target_directory", type=str, required=True, help="Target directory to move and rename the TSV files")
    args = parser.parse_args()

    collate_results(
        main_directory=args.main_directory,
        target_directory=args.target_directory
    )