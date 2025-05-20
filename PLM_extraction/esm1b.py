from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')
model = AutoModel.from_pretrained('facebook/esm1b_t33_650M_UR50S')

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

sequences = read_fasta(file_path)

embeddings = []

for label, sequence in sequences.items():
    sequence = ' '.join(sequence)
    inputs = tokenizer(sequence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state

    sequence_embedding = torch.mean(embedding, dim=1).squeeze().numpy()

    embeddings.append([label] + sequence_embedding.tolist())

df = pd.DataFrame(embeddings)

df.columns = ['name'] + [f'embedding_{i}' for i in range(1, df.shape[1])]
df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)

csv_file_path = './CPP_embedding_train/esm1b.csv'
df.to_csv(csv_file_path, index=False)

print(f"Embedding results saved to {csv_file_path}")
