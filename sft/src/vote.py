import pandas as pd
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import os


class ScoreRanker:
    def __init__(self):
        self.scores: dict[str, float] = {}

    def add_score(self, answer: str, weight: float):
        if answer in self.scores:
            self.scores[answer] += weight
        else:
            self.scores[answer] = weight

    def get_highest_score_answer(self):
        return max(self.scores, key=self.scores.get)

    def __repr__(self) -> str:
        return str(self.scores)


@dataclass
class VoteData:
    id: str
    score_ranker: ScoreRanker


def create_vote_data(files: list[str], weights: list[float]) -> list[VoteData]:
    dfs = [pd.read_csv(file) for file in files]
    vote_data = []
    for df_index, df in enumerate(dfs):
        for row_index, row in enumerate(df.iterrows()):
            _, row = row
            index = str(row['id'])
            if df_index == 0:
                vote_data.append(
                    VoteData(id=index, score_ranker=ScoreRanker()))
            vote_data[row_index].score_ranker.add_score(
                str(row['answer']), weights[df_index])
    return vote_data


def write_csv_file(vote_data: list[VoteData], output_file: str):
    dirname = os.path.dirname(output_file)
    os.makedirs(dirname, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('id,answer\n')
        for vote in vote_data:
            f.write(f'{vote.id},{vote.score_ranker.get_highest_score_answer()}\n')


def main(args: Namespace):
    print(args.weights)
    vote_datas = create_vote_data(args.vote_files, args.weights)
    print(vote_datas[200])
    write_csv_file(vote_datas, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vote_files", nargs="+",
                        help="List of files to vote")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="List of weights for each file")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
