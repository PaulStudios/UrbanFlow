import argparse
import subprocess

import uvicorn

from route_prediction_ai import download_data, main as route_main


def runserver():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)


def download_data_func():
    download_data.run()


def train_and_evaluate(download=False):
    if download:
        download_data_func()
    route_main.main()


def run_tests(test_name=None):
    tests = {
        "server": {"name": "server", "file": "server/tester.py", "pytest": False}
    }
    if test_name:
        if test_name in tests:
            test = tests[test_name]
            if test["pytest"]:
                subprocess.run(["pytest", test["file"]])
            else:
                subprocess.run(["python", test["file"]])
        else:
            print(f"No test found with name '{test_name}'")
    else:
        # Run all tests
        for test in tests.values():
            if test["pytest"]:
                subprocess.run(["pytest", test["file"]])
            else:
                subprocess.run(["python", test["file"]])


def main():
    parser = argparse.ArgumentParser(
        description="UrbanFlow Management Interface - Control hub for the UrbanFlow project."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Define runserver command
    subparsers.add_parser("runserver", help="Run the server.")

    # Define download_data command
    subparsers.add_parser("download_data", help="Download data.")

    # Define train_and_evaluate command with optional --download argument
    train_parser = subparsers.add_parser("train_and_evaluate", help="Train and evaluate the model.")
    train_parser.add_argument("--download", action="store_true", help="Download data before training.")

    # Define test command with optional test name argument
    test_parser = subparsers.add_parser("test", help="Run tests.")
    test_parser.add_argument("--test_name", type=str, help="Name of the test to run (e.g., --server).")

    args = parser.parse_args()

    if args.command == "runserver":
        runserver()
    elif args.command == "download_data":
        download_data_func()
    elif args.command == "train_and_evaluate":
        train_and_evaluate(download=args.download)
    elif args.command == "test":
        run_tests(test_name=args.test_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()