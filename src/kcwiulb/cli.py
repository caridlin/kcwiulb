import argparse

def main():
    parser = argparse.ArgumentParser(prog="kcwiulb")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print("kcwiulb 0.1.0")