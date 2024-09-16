import argparse
from script.process import process

def main():
    """
    Entry point for the ATPP command-line interface.
    STILL INCOMPLETE

    This function initializes the argument parser, defines subcommands and their arguments,
    and processes the provided command-line arguments. It supports the following subcommands:

    - `process`: Processes thermography data with optional filtering, visualization, and frequency settings.

    Usage:
        python cli.py process <input_data> [--filter <filter>] [--visualize <True|False>] [--frequency <frequency>]

    Arguments:
        input_data (str): Path to input thermography data.
        --filter (str, optional): Apply a filter (e.g., noise_reduction).
        --visualize (bool, optional): Enable visualization (default: False).
        --frequency (float, optional): Frequency for lock-in amplifier processing.

    If no command is provided, the help message is displayed.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="ATPP: Active Thermography Test Post-Processing Tool"
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help="Available commands")

    # Add a subparser for the `process` command
    process_parser = subparsers.add_parser('process', help="Process thermography data")
    process_parser.add_argument('input_data', type=str, help="Path to input thermography data")
    process_parser.add_argument('--filter', type=str, help="Apply a filter (e.g., noise_reduction)")
    process_parser.add_argument('--visualize', type=bool, default=False, help="Enable visualization (default: False)")
    process_parser.add_argument('--frequency', type=float, help="Frequency for lock-in amplifier processing")

    # Parse the arguments
    args = parser.parse_args()

    # If no command is provided, display help
    if not args.command:
        parser.print_help()
    elif args.command == 'process':
        # Call the `process` function with the provided arguments
        process(input_data=args.input_data, filter=args.filter, visualize=args.visualize, frequency=args.frequency)

if __name__ == "__main__":
    main()
