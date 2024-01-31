from collections import OrderedDict
from pyvi import ViTokenizer


def corpus_reader(path, delim='\t', word_idx=0, label_idx=-1):
    tokens, labels = [], []
    tmp_tok, tmp_lab, tmp_img = [], [], []
    label_set = []
    with open(path, 'r', encoding='utf8') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            imgid = ''
            if line.startswith('IMGID:'):
                imgid = line.strip()
                tmp_img.append(imgid)
                continue
            if len(cols) < 2:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok)
                    labels.append(tmp_lab)
                tmp_tok = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
    return tmp_img, tokens, labels, list(OrderedDict.fromkeys(label_set))


def simplify_labels(labels):
    if not labels:
        return None

    # Check if all labels are 'O'
    if all(label == 'O' for label in labels):
        return 'O'

    # Check for 'B-' or 'I-' prefixes and return the corresponding label
    for label in labels:
        if label.startswith('B-') or label.startswith('I-'):
            return label

    # If no 'B-' or 'I-' prefixes are found, return the first label
    return labels[0]


def align_labels(token1, label1, token2):
    label2 = []

    i = 0  # Pointer token1
    j = 0  # Pointer token2
    flag_B = False
    while j < len(token2):
        try:
            if token2[j] == token1[i]:
                flag_B = False
                # If tokens match, copy label from label1 to label2
                label2.append(label1[i])
                i += 1
                j += 1
            else:
                j += 1
                if label1[i] != 'O':
                    tmp_label = label1[i]
                    if label1[i].startswith('B-') and flag_B:
                        flag_B = False
                        tmp_label = 'I-' + tmp_label[2:]
                    flag_B = True
                    label2.append(tmp_label)
                    try:
                        if token2[j] == token1[i + 1]:
                            i += 1
                    except:
                        continue
                else:
                    flag_B = False
                    label2.append(label1[i])
                    try:
                        if token2[j] == token1[i + 1]:
                            i += 1
                    except:
                        continue
        except:
            #Check
            print("Warning: Check None in file exported")
            break
    return label2


if __name__ == '__main__':
    imgs, tokens, labels, a = corpus_reader(r'modules/datasets/vlsp2021/test.txt')

    with open('test2.txt', 'w', encoding='utf8') as f:
        for i, item in enumerate(zip(tokens, labels, imgs)):
            token, label, img = item[0], item[1], item[2]
            # change tokenize
            new_token = ViTokenizer.tokenize(' '.join(token)).split()
            re_split_space_token = [item for token_ in new_token for item in token_.split("_")]

            new_labels = align_labels(token, label, re_split_space_token)

            # Solve other case: len(token2) < len(token1)
            f.write(str(img))
            f.write('\n')
            index_label = 0
            for new_t in new_token:
                num_label_stack = len(new_t.split("_"))
                new_label = new_labels[index_label:index_label + num_label_stack]
                index_label += num_label_stack
                f.write(str(new_t) + '\t' + str(simplify_labels(new_label)))
                f.write('\n')
            f.write('\n')
