import os
import glob

import numpy as np
import pandas as pd

def find_submission_files():
    folder = os.path.join("models", "*/*submission*")
    print(folder)
    files = [
        file_path
        for file_path
        in glob.iglob(folder, recursive=True)
    ]
    return files

def main():
    submissions = find_submission_files()
    print(submissions)
    dfs = []
    for i, sub in enumerate(submissions):
        dfs.append(pd.read_csv(sub))
    columns = list(filter(lambda a: a != 'Article', list(dfs[0])))

    submission = pd.DataFrame(dfs[0]['Article'])
    for column in columns:
        slices = []
        for df in dfs:
            slices.append(df[column].values)
        submission[column] = np.average(slices, axis=0, weights=[3./4, 1./4])
    print(submission)
    submission.to_csv('ensemble_submission.csv', index=False)


if __name__ == '__main__':
    main()