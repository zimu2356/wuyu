"""Restore portable relative references for the frozen Whole 1.1 snapshot."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'experiments/relation_start_snapshot'

def main():
    for name in ('source_projects','reference_runtime_v1'):
        destination=SNAPSHOT/name
        expected=ROOT/name
        if not expected.is_dir():raise FileNotFoundError(expected)
        if destination.is_symlink() or destination.exists():
            if destination.resolve()!=expected.resolve():raise FileExistsError(destination)
        else:destination.symlink_to(Path('../..')/name,target_is_directory=True)
    print('Frozen snapshot references restored; current runtime is unchanged.')

if __name__=='__main__':main()
