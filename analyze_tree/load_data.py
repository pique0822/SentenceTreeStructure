def parse_line(line, type):
    sentence, tree_struct, open_nodes, adj_nodes = line.split('|')
    if type == 'open_nodes':
        rel = open_nodes.lstrip().strip().replace("\\","")
    else:
        rel = adj_nodes.lstrip().strip().replace("\\","")
    rel = rel.replace("n","")
    rel = rel.replace("'","")
    brackets_open = [int(x) for x in rel.split()]
    return sentence.lstrip().strip(), brackets_open

input_path = 'dahaene_dataset/filtered_sentences.txt'

def load_data(input_path):
    sentences = []
    labels = []
    with open(input_path) as input_file:
        for line in input_file:
            sentence, label = parse_line(line,'open_nodes')
            sentences.append(sentence)
            labels.append(label)
    return sentences,labels
