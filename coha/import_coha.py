import os
import re
import json
from glob import glob
from collections import Counter
from optparse import OptionParser

from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--output-subdir', type=str, default='clean',
                      help='Output subdir: default=%default')
    parser.add_option('--decades', type=str, default='1960,1990',
                      help='Only select certain decades (comma-separted): default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    output_subdir = options.output_subdir
    decades = options.decades

    raw_dir = os.path.join(basedir, 'raw')
    if not os.path.exists(raw_dir):
        print("Directory not found:", raw_dir)
        raise FileNotFoundError

    outdir = os.path.join(basedir, output_subdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    html_counter = Counter()
    chunk_lengths = Counter()

    decades = [int(d) for d in decades.split(',')]

    outlines = []
    for decade in decades:
        if decade < 1980:
            group = 'pre1980s'
        else:
            group = 'post1970s'
        indir = os.path.join(raw_dir, str(decade) + 's')
        files = sorted(glob(os.path.join(indir, '*.txt')))
        print(indir, len(files))
        total_chunks = 0
        for file_i, infile in enumerate(tqdm(files)):
            basename = os.path.basename(infile)[:-4]
            try:
                assert len(basename.split('_')) == 3
            except AssertionError as e:
                print(basename)
                raise e
            subset, year, doc_id = basename.split('_')
            with open(infile) as f:
                lines = f.readlines()
            skip = False
            if len(lines) == 3:
                assert lines[0].startswith('@@')
                assert lines[1] == '\n'
                assert len(lines[2]) > 3
                text = lines[2].strip()
            elif len(lines) == 1:
                text = ' '.join(lines[0].split()[1:]).strip()
            else:
                print("Skipping", infile)
                skip = True
                text = None

            if not skip:
                tokens = text.split()
                # remove HTML tags
                html_tokens = [t for t in tokens if t.startswith('<')]
                html_counter.update(html_tokens)
                tokens = [t for t in tokens if not t.startswith('<')]
                text = ' '.join(tokens)
                
                # Do some minor replacements
                text = re.sub('##+', ' ', text)
                text = re.sub('_', ' ', text)
                
                # split on the redacted tokens
                chunks = text.split('@ @ @ @ @ @ @ @ @ @')
                total_chunks += len(chunks)
                for c_i, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    chunk_lengths[len(chunk)] += 1
                    # discard very short chunks
                    if len(chunk) > 8:
                        outlines.append({'id': basename + '_' + str(c_i).zfill(5),
                                         'decade': decade,
                                         'year': int(year),
                                         'subset': subset,
                                         'group': group,
                                         'text': chunk,
                                         })

    with open(os.path.join(outdir, 'all.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    print("Chunk lengths")
    for c in range(10):
        print(c, chunk_lengths[c])
    max_length = max(chunk_lengths)
    print(max_length, chunk_lengths[max_length])

    print("HTML tags")
    for t, c in html_counter.most_common():
        print(t, c)


if __name__ == '__main__':
    main()
