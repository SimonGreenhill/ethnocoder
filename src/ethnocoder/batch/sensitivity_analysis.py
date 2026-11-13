# For the temperature argument, pick several values, e.g. 0.2, 0.4, 0.6, 0.8, 1.0.
# Call the batch_and_summarise main method once per temperature value.
# Run a final meta-summary, comparing and plotting results per temperature value.

import os
import argparse
import copy
from rich import print
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ethnocoder.common.utils import get_timestamp, GRAMBANK_CLDF_DIR, CONTEXT_BUDGET_DEFAULT, is_glottocode
from ethnocoder.batch.batch_and_summarise import main as batch_and_summarise_main

def plot_metasummary(
        results_dict: dict,
        metasummary_dir: str,
) -> None:
    """
    Plot accuracy vs time/tokens results for different temperature values.

    Output fig to metasummary_dir as PNG.

    Args:
        results_dict (dict): Dictionary containing results per temperature value.
        metasummary_dir (str): Directory to save the output plot as PNG.
    """

    # Create plot
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Time (s)")
    # ax2.set_ylabel("Reasoning (1000x tokens)")

    # Loop at the different temperatures in results_dict;
    # we already have colours as glottocodes
    # so we need to differentiate the different temperature values by shape.
    # plot with different markers or line styles for each temperature value.

    # How many temperatures do we have?
    n_temperatures = len(results_dict)

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+"]

    if len(markers) < n_temperatures:
        raise ValueError("Not enough markers for the number of temperature values.")
    markers = markers[:n_temperatures]

    j = 0
    for temp, results in results_dict.items():

        # Plot accuracy on the x-axis (scale 0 to 100)
        total_elapsed = [v["resources"]["total_time"] for v in results.values()]


        # Plot each subdirectory's data points with unique colors and markers.
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        subdirectory_handles = []

        # Plot in a loop so we get the elapsed and the reasoning in the same colour, different shapes.
        for i, (basename, values) in enumerate(sorted(results.items())):
            color = colors[i % len(colors)]
            accuracy = values["totals"]["correct_pct"]

            ax1.plot(
                accuracy, 
                values["resources"]["total_time"],
                marker=markers[j], 
                linestyle="None", 
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

            # Legend should take the first marker for each subdirectory, not the current temperature's marker.
            subdirectory_handles.append(
                Line2D(
                    [], 
                    [], 
                    color=color, 
                    marker=markers[0], 
                    linestyle="None", 
                    label=basename,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                )
            )
        j += 1

    ax1.set_xlim(0, 100)

    subdirectory_legend = ax1.legend(
        handles=subdirectory_handles, title="Glottocode", loc="upper left"
    )
    ax1.add_artist(subdirectory_legend)

    # Make the total time axis go from 0 to 1.1* the max
    ax1.set_ylim(0, 1.1 * max(total_elapsed))

    # Set x-ticks to have the percentage sign
    ax1.set_xticks([tick for tick in ax1.get_xticks() if tick % 10 == 0])
    ax1.set_xticklabels([f"{int(tick)}%" for tick in ax1.get_xticks()])

    # Legend for temperature markers.
    temperature_handles = [
        Line2D([], [], color="black", marker=markers[i], linestyle="None", label=list(results_dict.keys())[i])
        for i in range(len(results_dict.keys()))
    ]
    ax1.legend(handles=temperature_handles, title="Temperature", loc="lower left")

    # 5. Output to the summary subdirectory
    os.makedirs(metasummary_dir, exist_ok=True)
    output_path = os.path.join(metasummary_dir, f"accuracy_time_reasoning_{get_timestamp()}.png")
    plt.savefig(output_path)
    print(f"[green]Saved plot to [spring_green2]{output_path}[/spring_green2][/green]")

def metasummarise(
        output_batch_directory: str,
)-> None:
    """
    Perform a meta-summary over the results of different temperature values.

    For every subdirectory of output_batch_directory, get the temperature value.
    These are the keys of the ultimate results dict.

    For each of these, get the stats per glottocode.
    Store them in the inner dict, per temp.

    So we have:
    {
        "temp_0.2": {
            "glottocode_1": {
                  "resources": {
                      "elapsed_1": 177.20000000000002,
                      "elapsed_2": 0.0,
                      "total_time": 177.20000000000002,
                      "total_reasoning_tokens": 44861
                  },
                     "totals": {
                          "correct": 162,
                          "incorrect": 32,
                          "hedged": 3,
                          "correct_pct": 82.2,
                          "incorrect_pct": 16.2,
                          "hedged_pct": 1.5
                      }
                  },
            "glottocode_2": {
                  "resources": {...}
                  "totals": {...}
                }
            ...
    },
            ...
        },
        "temp_0.4": {
            "glottocode_1": {...},
            "glottocode_2": {...},
            ...
        },
        ...
    }

    Args:
        output_batch_directory (str): The directory containing subdirectories for each temperature value.
    """

    print(f"[blue]Starting meta-summary for output batch directory:[/blue] [bright_cyan]{output_batch_directory}[/bright_cyan]")

    # 1. Loop at subdirectories of output_batch_directory
    # 2. Extract temperature value from subdirectory name
    # 3. Collect stats per glottocode for each temperature
    # 4. Store in results dict keyed by temperature
    results_dict = {}

    for subdir in os.listdir(output_batch_directory):

        # Get the subdirectory path and check if it represents a temperature value
        subdir_path = os.path.join(output_batch_directory, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith("temp_"):

            # Get the temp
            temp = subdir.split("_")[1]

            print(f"[blue]Processing temperature: [bold]{temp}[/bold][/blue]")

            # Initialise data for this temperature
            results_dict[temp] = {}

            for glottocode_dir in os.listdir(subdir_path):

                # Skip the "summary" subdirectory
                if glottocode_dir == "summary":
                    continue

                # Check the subdir has exactly 8 characters
                if not is_glottocode(glottocode_dir):
                    print(f"  [yellow]Warning: Skipping invalid glottocode directory: {glottocode_dir}[/yellow]")
                    continue

                # Get the exact path to this glottocode under this temperature for this batch.
                glottocode_path = os.path.join(subdir_path, glottocode_dir)
                if os.path.isdir(glottocode_path):
                    # Load stats from a file, e.g., stats.json
                    stats_file = os.path.join(glottocode_path, "stats.json")
                    if os.path.exists(stats_file):
                        with open(stats_file, "r") as f:
                            results_dict[temp][glottocode_dir] = json.load(f)

                        print(f"  [green]Loaded stats for glottocode: [bold]{glottocode_dir}[/bold][/green]")
                    else:
                        print(f"  [yellow]Warning: Stats file not found for glottocode: {glottocode_dir}[/yellow]")
        
        print(f"[green]Finished processing temperature: [bold]{temp}[/bold][/green]")

    # Sort the dict by keys as floats, ascending.
    results_dict = dict(sorted(results_dict.items(), key=lambda x: float(x[0])))

    # Now dump total results to a JSON file in <metasummary> subdirectory of the output batch directory
    metasummary_dir = os.path.join(output_batch_directory, "metasummary")
    os.makedirs(metasummary_dir, exist_ok=True)
    summary_file = os.path.join(metasummary_dir, "metasummary.json")
    with open(summary_file, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"[green]Meta-summary saved to: [spring_green2]{summary_file}[/spring_green2][/green]")

    # Plot results.
    plot_metasummary(
        results_dict=results_dict,
        metasummary_dir=metasummary_dir,
    )

def main(
        args: argparse.Namespace,
)->str:
    """
    Call batch_and_summarise_main() for each temperature value specified in the arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing temperature values and other options.
    """

    # Get the different temperature values.
    temperatures = [float(t) for t in args.temperatures.split(",")]

    # Remove temperatures from args
    del args.temperatures

    # If there is no output batch directory specified, create one under the master output directory
    if args.output_batch_directory is None:

        # For storing results, get the basename of the input directory
        input_basename = os.path.basename(os.path.normpath(args.input_directory))

        # Create master output directory, create subdirectories per temperature value
        ts = get_timestamp()
        output_batch_directory = os.path.join(
            "output", # master output
            "batches", # to help us separate batches from single runs
            f"{ts[:8]}_{input_basename}", # date and batch number
            f"{ts[9:]}_{args.model.replace('openai/','').replace('ollama/','')}", # time and model
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_batch_directory, exist_ok=True)

    else:
        output_batch_directory = args.output_batch_directory

    for temp in temperatures:

        # Create subdirectories per temperature value
        subdir_temp = os.path.join(output_batch_directory, f"temp_{temp}")
        os.makedirs(subdir_temp, exist_ok=True)

        # Add current temperature and output subdirectory to args.
        args.temperature = temp
        args.output_batch_directory = subdir_temp

        # Create an exact copy of args, because batch_and_summarise_main() might modify it
        args_copy = copy.deepcopy(args)

        # Call batch_and_summarise_main() with the current temperature and output subdirectory
        batch_and_summarise_main(
            args = args_copy,
        )

    # Meta-summary over temperatures.
    metasummarise(
        output_batch_directory = output_batch_directory,
    )

    return output_batch_directory


if __name__ == "__main__":

    # Arguments

    parser = argparse.ArgumentParser(
        description="Prepare and run sensitivity analysis (currently only temperature) for batch processing."
    )

    # Argument to ONLY run metasummarise, not main.
    parser.add_argument(
        "-mo",
        "--metasummarise-only",
        action="store_true",
        help="Only run metasummarise, not the main batch processing."
    )

    # Temperature values for sensitivity analysis
    parser.add_argument(
        "-t",
        "--temperatures",
        type=str,
        required=False,
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated list of temperature values (default: 0.2,0.4,0.6,0.8,1.0)"
    )
    
    # Directories
    parser.add_argument(
        "-i", 
        "--input_directory", 
        type=str, 
        help="Path to the input directory containing subdirectories with documents and gold-coding JSON files.",
    )

    # Output batch directory
    parser.add_argument("-o", "--output-batch-directory", default=None, help="Path to the output batch directory for this run. If None, a new directory will be created.")

    # NB DO NOT use -p as it clashes with -p in code.py, which this script passes args to.
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the input data without running the batch.")

    # model: the LLM to use for coding the document
    parser.add_argument(
        "-m",
        "--model", 
        default=None,
        type=str,
        help="LiteLLM model string",
    )

    # variables: CSV file containing the variables to code
    parser.add_argument("--variables", default="parameters.csv", help="Variables CSV (default: parameters.csv)")

    # codes: CSV file containing the codes for the variables
    parser.add_argument("--codes", default="codes.csv", help="Codes CSV (default: codes.csv)")

    # ids: Comma-separated list of variable IDs to code
    parser.add_argument("--ids", help="Comma-separated list of variable IDs to code (e.g. 2,3,5)")

    # api-base: Override API base URL for the LLM
    parser.add_argument(
        "--api-base", 
        default=None,
        help="Override API base URL",
    )

    # max-chars: Truncate document text to this many characters
    parser.add_argument("--max-chars", type=int, default=None, help="Truncate document text to this many characters")

    # context-budget: Token budget per request
    parser.add_argument(
            "--context-budget",
            type=int,
            default=CONTEXT_BUDGET_DEFAULT,
            help="Token budget per request. When set, variables are coded in batches that fit "
            "this budget (reusing a cached document prefix). Omit for a single request with all variables.",
        )
    
        # response-reserve-per-var: Estimated output tokens reserved per variable when packing batches
    parser.add_argument(
            "--response-reserve-per-var",
            type=int,
            default=500,
            help="Estimated output tokens reserved per variable when packing batches (default: 500)",
        )

    # bool-review: Perform a review of the coding results where necessary
    parser.add_argument("--bool-review", action="store_true", help="Perform a review of the coding results where necessary")
    
        # bool-thinking: Enable thinking mode for the LLM, if allowed by the model
    parser.add_argument("--bool-thinking", action="store_true", help="Enable thinking mode for the LLM, if allowed by the model")
    
    # top-p: Set the top-p (nucleus sampling) for the LLM
    # Default is 0.95, recommended by SAIA for GLM 5.3 Flash.
    parser.add_argument("--top-p", type=float, default=0.95, help="Set the top-p (nucleus sampling) for the LLM")

    args = parser.parse_args()

    if args.metasummarise_only:

        # args must contain output_batch_directory
        output_batch_directory = args.output_batch_directory

        if not output_batch_directory:
            raise ValueError("output_batch_directory must be specified when using --metasummarise-only")

        metasummarise(
            output_batch_directory = output_batch_directory,
        )
    else:
        output_batch_directory = main(args=args)
        metasummarise(
            output_batch_directory = output_batch_directory,
        )
