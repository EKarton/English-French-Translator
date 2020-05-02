import os
import string

from nltk.tokenize import word_tokenize


def get_size_of_corpus(filepaths):
    """ Given a list of filepaths, it will return the total number of lines

        Parameters
        ----------
        filepaths : [ str ]
            A list of filepaths

        Returns
        -------
        num_lines : int
            The total number of lines in filepaths
    """

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    num_lines = 0
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            num_lines += sum(bl.count("\n") for bl in blocks(f))

    return num_lines


def get_tokens_from_line(line):

    # Make the text lowercase and remove the \n
    line = line.lower().strip()

    # Standardize single quotations
    line = (
        line.replace("\u2019", "'")
        .replace("\u0060", "'")
        .replace("\u00B4", "'")
        .replace("\u2018", "'")
        .replace("\u201A", "'")
        .replace("\u201B", "'")
        .replace("\u2039", "'")
        .replace("\u203A", "'")
        .replace("\u275B", "'")
        .replace("\u275C", "'")
        .replace("\u276E", "'")
        .replace("\u276F", "'")
    )

    # Standardize dashes
    line = (
        line.replace("\u00AD", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2015", "-")
        .replace("\u2212", "-")
        .replace("\u02D7", "-")
    )

    # Standardize double quotations
    line = (
        line.replace("\u201C", '"')
        .replace("\u201D", '"')
        .replace("\u2033", '"')
        .replace("\u00AB", '"')
        .replace("\u00BB", '"')
        .replace("\u201E", '"')
        .replace("\u201F", '"')
        .replace("\u275D", '"')
        .replace("\u275E", '"')
        .replace("\u301D", '"')
        .replace("\u301E", '"')
        .replace("\u301F", '"')
        .replace("\uFF02", '"')
    )

    # Standardize the ellipses
    line = line.replace("\u2026", "...")

    # Tokenize the text
    tokens = word_tokenize(line)

    # Remove tokens that are spaces, tabs, newlines, etc
    tokens = list(filter(lambda token: not token in string.whitespace, tokens))    

    # Replace tokens that contain numbers with 'NUM'
    num_vals = []
    new_tokens = []
    for token in tokens:
        new_token = token

        if any(char.isdigit() for char in token):
            new_token = "NUM"
            num_vals.append(token)

        new_tokens.append(new_token)
    
    tokens = new_tokens

    return tokens, num_vals


def read_transcription_files(filepaths):
    """ Generate line info from data in a file for a given language

        Parameters
        ----------
        lang : {'en', 'fr'}
            Whether to tokenize the English sentences ('e') or French ('f').
        filenames : sequence
            Only tokenize sentences with matching names. If :obj:`None`, searches
            the whole directory in C-sorted order.

        Yields
        ------
        tokenized, filename, offs : list
            `tokenized` is a list of tokens for a line. `filename` is the source
            file. `offs` is the start of the sentence in the file, to seek to.
            Lines are yielded by iterating over lines in each file in the order
            presented in `filenames`.
    """
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            offs = f.tell()
            line = f.readline()
            while line:
                tokens, _ = get_tokens_from_line(line)

                yield tokens, filepath, offs
                offs = f.tell()
                line = f.readline()


def get_parallel_text(dir_, langs):
    """ Get a list of all files in 'dir_' with a file extension that is in 'langs'

        Parameters
        ----------
        dir_ : str
            A path to the transcription dictionary
        langs : [str]
            A list of file extensions for the parallel texts

        Returns
        -------
        filenames : list
            A list of all parallel texts' file names without the file extension
    """
    # Get a set of files that has the lang file extension
    files = os.listdir(dir_)
    filenames = []
    for lang in langs:
        lang = "." + lang
        lang_filenames = set(
            filename[:-3] for filename in files if filename.endswith(lang)
        )
        filenames.append(lang_filenames)
    del files

    if len(filenames) == 0:
        raise ValueError(
            f"Directory {dir_} contains no transcriptions ending in {lang} "
        )

    # Get the set of files that has the same name
    transcriptions_filenames = filenames[0]
    for lang_filenames in filenames:
        transcriptions_filenames = transcriptions_filenames & lang_filenames

    transcriptions_filenames = sorted(transcriptions_filenames)

    if len(transcriptions_filenames) == 0:
        raise ValueError(
            f"Directory {dir_} contains no common files ending in {lang}."
            f"Are you sure this is the right directory?"
        )

    # Create the output
    parallel_text = []
    for filename in transcriptions_filenames:
        parallel_text.append(tuple([filename + "." + lang for lang in langs]))

    return parallel_text
