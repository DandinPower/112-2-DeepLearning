import pandas as pd
from collections import Counter
from argparse import ArgumentParser, Namespace


def vote(files: list[str]):
    print(files)
    # Read all files into dataframes
    dfs = [pd.read_csv(file) for file in files]
    # Concatenate all dataframes
    df = pd.concat(dfs)
    # Group by index and apply voting function to rating
    result = df.groupby('id')['answer'].apply(
        lambda x: Counter(x).most_common(1)[0][0])
    return result


def main(args: Namespace):
    vote_csv = vote(args.vote_files)
    with open(args.output_file, 'w') as f:
        f.write('id,answer\n')
        for index, rating in vote_csv.items():
            f.write(f'{index},{rating}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vote_files", nargs="+",
                        help="List of files to vote")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
