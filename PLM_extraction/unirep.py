import torch
import pandas as pd
from tape import ProteinBertModel, TAPETokenizer

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        current_label = None
        current_sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_label:
                    sequences[current_label] = ''.join(current_sequence)
                current_label = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_label:
            sequences[current_label] = ''.join(current_sequence)
    return sequences

file_path = './data/CPP924_train.txt'

tokenizer = TAPETokenizer(vocab='unirep')
model = ProteinBertModel.from_pretrained('bert-base')

sequences = read_fasta(file_path)

embeddings = []

for label, sequence in sequences.items():
    inputs = torch.tensor([tokenizer.encode(sequence)])
    with torch.no_grad():
        outputs = model(inputs)
    embedding = outputs[0].mean(dim=1).squeeze().numpy()
    embeddings.append([label] + embedding.tolist())

df = pd.DataFrame(embeddings)

df.columns = ['name'] + [f'embedding_{i}' for i in range(1, df.shape[1])]
df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)

csv_file_path = './CPP_embedding_train/bert_unirep.csv'
df.to_csv(csv_file_path, index=False)

print(f"Embedding results saved to {csv_file_path}")