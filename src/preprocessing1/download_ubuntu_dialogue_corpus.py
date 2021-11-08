import argparse
import os
import tarfile
import urllib

import numpy as np
import pyprind

import utils


"""
次の理由から、本家の```create_ubuntu_dataset.py```の代わりにこちらを使う。
- ubuntu_dialogs.tgzはサイズが莫大で、また今回に関してはそのすべては必要ではないため、解凍せずに条件にあうものだけを抽出する
- 今回に関しては (context, response, label) のtripletsの生成までは不要であり、条件にあうdialogだけが必要
"""


URL = 'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
ARCHIVE_NAME = 'ubuntu_dialogs.tgz'


def main(args):
    archive_dir = args.input
    output_dir = args.output

    # Download archived dialogues (```ubuntu_dialogs.tgz```) to ```archive_dir```
    prepare_data_maybe_download(archive_dir)

    # Extract dialogues that meet the given conditions
    dialogues = extract_dialogues(archive_path=os.path.join(archive_dir, ARCHIVE_NAME),
                                  n_dialogues=args.n_dialogues,
                                  min_dialogue_length=args.min_dialogue_length,
                                  max_dialogue_length=args.max_dialogue_length,
                                  max_utterance_length=args.max_utterance_length,
                                  max_speakers=args.max_speakers)
    assert len(dialogues) <= args.n_dialogues

    # Save the extracted dialogues to ```output_dir```
    save_dialogues(output_dir=output_dir, dialogues=dialogues)

    utils.writelog("Done.")


def prepare_data_maybe_download(archive_dir):
    """
    Download archived dialogues if necessary.
    This functions is mainly copied from the following original repository:
        https://github.com/rkadlec/ubuntu-ranking-dataset-creator
    """
    # Check
    filenames = os.listdir(archive_dir)
    assert "generate.sh" in filenames
    assert "create_ubuntu_dataset.py" in filenames
    assert "download_punkt.py" in filenames
    assert "meta" in filenames

    # dialogs are missing
    archive_path = os.path.join(archive_dir, ARCHIVE_NAME)
    if not os.path.exists(archive_path):
        # archive missing, download it
        utils.writelog("Downloading %s to %s" % (URL, archive_path))
        filepath, _ = urllib.request.urlretrieve(URL, archive_path)
        utils.writelog("Successfully downloaded " + filepath)
    else:
        utils.writelog("Found archive: %s" % archive_path)


def extract_dialogues(
        archive_path,
        n_dialogues,
        min_dialogue_length,
        max_dialogue_length,
        max_utterance_length,
        max_speakers):
    utils.writelog("Number of dialogues: %d" % n_dialogues)
    utils.writelog("Min. dialogue length: %d" % min_dialogue_length)
    utils.writelog("Max. dialogue length: %d" % max_dialogue_length)
    utils.writelog("Max. utterance length: %d" % max_utterance_length)
    utils.writelog("Max. speakers: %d" % max_speakers)

    utils.writelog("Extracting dialogues from %s ..." % archive_path)
    dialogues = []

    with tarfile.open(name=archive_path, mode="r") as tar:
        # Get archived files (including directories)
        utils.writelog("Extracting archived information ...")

        members = tar.getmembers() # May take several minutes
        utils.writelog("Number of archived entries (files + directories): %d" % len(members))
        members = [m for m in members if m.name.endswith(".tsv")]
        utils.writelog("Number of archived TSV files: %d" % len(members))

        count = 0
        avg_dialogue_length = []
        avg_utterance_length = []
        avg_speakers = []
        for member_i, member in enumerate(members):
            # Content
            with tar.extractfile(member) as f:
                binary = f.read()
                text = binary.decode("utf-8")
            lines = text.split("\n")
            lines = [line.split("\t") for line in lines]

            # Clean lines
            new_lines = []
            for items in lines:
                assert len(items) == 4 or len(items) == 1 or len(items) == 0
                if len(items) == 4:
                    new_lines.append(items)
            lines = new_lines

            # Clean utterance
            lines = [items for items in lines if len(items) == 4]
            for i in range(len(lines)):
                assert len(lines[i]) == 4
                utterance = lines[i][3]
                utterance = utterance.strip()
                lines[i][3] = utterance

            # If conditions are met, record this dialogue
            avg_dialogue_length.append(len(lines))
            if min_dialogue_length <= len(lines) <= max_dialogue_length:
                # Dialogue length is OK
                all_with_response = True
                for items in lines[2:]:
                    _, _, listener, _ = items
                    if listener == "":
                         all_with_response = False
                all_with_utterance = True
                for items in lines:
                    _, _, _, utterance = items
                    if utterance == "":
                        all_with_utterance = False
                if all_with_response and all_with_utterance:
                    # All utterances (except for the first one) are with response-to markers
                    temp_max_utterance_length = -1
                    speakers = []
                    for items in lines:
                        _, speaker, listener, utterance = items
                        n_tokens = len(utterance.split(" ")) # rough whitespace-based tokenization
                        temp_max_utterance_length = max(temp_max_utterance_length, n_tokens)
                        speakers.append(speaker)
                        speakers.append(listener)
                    speakers = set(speakers)
                    avg_utterance_length.append(temp_max_utterance_length)
                    avg_speakers.append(len(speakers))
                    if temp_max_utterance_length <= max_utterance_length and len(speakers) <= max_speakers:
                        # Utterance length and the number of speakers are OK
                        dialogues.append(lines)
                        count += 1
                        # Progress
                        if count % 1000 == 0:
                            utils.writelog("##### Extracted %d dialogues #####" % count)
                        if count == n_dialogues:
                            break

            # Progress
            if (member_i + 1) % 5000 == 0:
                utils.writelog("Processed %d dialogues" % (member_i + 1))
                utils.writelog("Avg. dialogue length: %f" % np.mean(avg_dialogue_length))
                utils.writelog("Avg. max utterange length: %f" % np.mean(avg_utterance_length))
                utils.writelog("Avg. number of speakers: %f" % np.mean(avg_speakers))
                avg_dialogue_length = []
                avg_utterance_length = []
                avg_speakers = []

    ratio = float(count) / len(members) * 100.0
    utils.writelog("Extracted %d dialogues (utility: %d/%d=%.2f%%)" % (count, count, len(members), ratio))
    return dialogues


def save_dialogues(output_dir, dialogues):
    utils.writelog("Saving dialogues to %s ..." % output_dir)
    utils.mkdir(output_dir)
    for dialogue_i, dialogue in enumerate(pyprind.prog_bar(dialogues)):
        with open(os.path.join(output_dir, "%06d.tsv" % dialogue_i), "w") as f:
            for items in dialogue:
                assert len(items) == 4
                line = "\t".join(items)
                f.write("%s\n" % line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_dialogues", type=int, required=True)
    parser.add_argument("--min_dialogue_length", type=int, default=7)
    parser.add_argument("--max_dialogue_length", type=int, default=16)
    parser.add_argument("--max_utterance_length", type=int, default=20)
    parser.add_argument("--max_speakers", type=int, default=9)
    args = parser.parse_args()
    main(args)

