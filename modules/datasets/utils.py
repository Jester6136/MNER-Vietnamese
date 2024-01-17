from collections import OrderedDict

def corpus_reader(path, delim='\t', word_idx=0, label_idx=-1):
    tokens, labels = [], []
    tmp_tok, tmp_lab = [], []
    label_set = []
    with open(path, 'r',encoding='utf8') as reader:
        for line in reader:
            line = line.strip()
            cols = line.split(delim)
            if len(cols) < 2:
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok); labels.append(tmp_lab)
                tmp_tok = []
                tmp_lab = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
    return tokens, labels, list(OrderedDict.fromkeys(label_set))

if __name__=='__main__':
    _,_,a=corpus_reader(r'modules/datasets/vlsp2016/test.txt')
    _,_,b=corpus_reader(r'modules/datasets/vlsp2016/train.txt')
    _,_,c=corpus_reader(r'modules/datasets/vlsp2016/dev.txt')
    all_entities = list(set(a + b + c))
    print('\n'.join(all_entities))
    print(len(all_entities))
