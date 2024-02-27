import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### PROVIDED CODE #####
def load_datasets(data_directory: str) -> "Union[dict, dict]":
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
        text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]


def build_index_map(
        word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words - 1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}


# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def tokens_to_ix(
        tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader:
    def __init__(self, filepath, seq_len, examples_per_epoch):
        # Load the dataset
        with open(filepath, 'r') as file:
            data = file.read()

        # Process the dataset into a list of characters
        self.data = list(data)

        # Set up unique characters
        self.unique_chars = list(set(self.data))

        # Generate character mappings
        self.mappings = self.generate_char_mappings(self.unique_chars)

        # Process the dataset into indices
        self.data_indices = self.convert_seq_to_indices(self.data)

        # Convert to tensor and transfer to GPU if available
        self.data_indices = torch.tensor(self.data_indices).to(device)

        # Initialize other instance variables
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch
        self.vocab_size = len(self.unique_chars)

    def generate_char_mappings(self, unique_chars):
        # Create character-to-index and index-to-character dictionaries
        char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        return {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}

    def convert_seq_to_indices(self, seq):
        # Convert a sequence of characters to indices
        return [self.mappings['char_to_idx'][char] for char in seq]

    def convert_indices_to_seq(self, seq):
        # Convert a sequence of indices to characters
        return [self.mappings['idx_to_char'][idx] for idx in seq]

    def get_example(self):
        # Generate examples for training
        for _ in range(self.examples_per_epoch):
            start_idx = torch.randint(0, len(self.data_indices) - self.seq_len - 1, (1,)).item()
            in_seq = self.data_indices[start_idx:start_idx + self.seq_len]
            target_seq = self.data_indices[start_idx + 1:start_idx + self.seq_len + 1]
            yield in_seq, target_seq


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars

        self.embedding_size = embedding_size

        # Initialize layers
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        self.wax = nn.Linear(embedding_size, hidden_size)
        self.waa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wya = nn.Linear(hidden_size, n_chars)

    def rnn_cell(self, i, h):
        h_new = torch.tanh(self.wax(i) + self.waa(h))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden=None):
        input_seq = self.embedding_layer(input_seq)
        if hidden is None:
            hidden = torch.zeros(1, self.hidden_size).to(input_seq.device)

        outputs = []
        for i in range(input_seq.size(0)):
            o, hidden = self.rnn_cell(input_seq[i].unsqueeze(0), hidden)
            outputs.append(o)
        out = torch.stack(outputs).squeeze(1)
        return out, hidden

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_p=None, top_k=None):
        generated_seq = [starting_char]
        hidden = torch.zeros(1, self.hidden_size).to(self.embedding_layer.weight.device)
        input_char = torch.tensor([starting_char], dtype=torch.long).to(self.embedding_layer.weight.device)
        input_char = self.embedding_layer(input_char)

        for _ in range(seq_len):
            output, hidden = self.rnn_cell(input_char, hidden)
            output = output.squeeze(0) / temp

            if top_k is not None:
                output = top_k_filtering(output, top_k)
            if top_p is not None:
                output = top_p_filtering(output, top_p)

            probs = F.softmax(output, dim=-1)
            next_char = Categorical(probs).sample()
            generated_seq.append(next_char.item())
            input_char = next_char.unsqueeze(0)
            input_char = self.embedding_layer(input_char)

        return generated_seq

    def get_loss_function(self):
        # Return the loss function
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        # Return the optimizer
        return torch.optim.Adam(self.parameters(), lr=lr)


class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        # Initialize layers
        self.embedding_layer = nn.Embedding(n_chars, embedding_size)
        self.forget_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.cell_state_layer = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, n_chars)

    def forward(self, input_seq, hidden=None, cell=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(input_seq.device)
        if cell is None:
            cell = torch.zeros(self.hidden_size).to(input_seq.device)

        outputs = []
        for i in range(input_seq.size(0)):
            embedded_input = self.embedding_layer(input_seq[i])  # No unsqueeze
            o, hidden, cell = self.lstm_cell(embedded_input, hidden, cell)
            outputs.append(o)
        out = torch.stack(outputs)
        return out, hidden, cell

    def lstm_cell(self, i, h, c):
        # Concatenate the input and hidden state
        combined = torch.cat((i, h), dim=0)  # Make sure the concatenation is along the correct dimension

        # Compute the gate activations and cell state updates
        forget = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        cell_candidate = torch.tanh(self.cell_state_layer(combined))

        # Update the cell state and hidden state
        c_new = forget * c + input_gate * cell_candidate
        h_new = output_gate * torch.tanh(c_new)

        # Compute the output
        o = self.fc_output(h_new)
        return o, h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        generated_seq = [starting_char]
        hidden = torch.zeros(self.hidden_size).to(self.embedding_layer.weight.device)
        cell = torch.zeros(self.hidden_size).to(self.embedding_layer.weight.device)
        input_char = torch.tensor(starting_char, dtype=torch.long).to(self.embedding_layer.weight.device)

        for _ in range(seq_len):
            output, hidden, cell = self.lstm_cell(self.embedding_layer(input_char), hidden, cell)
            output = output.squeeze(0) / temp

            if top_k is not None:
                output = top_k_filtering(output, top_k)
            if top_p is not None:
                output = top_p_filtering(output, top_p)

            probs = F.softmax(output, dim=-1)
            next_char = Categorical(probs).sample()
            generated_seq.append(next_char.item())
            input_char = next_char

        return generated_seq


def top_k_filtering(logits, top_k=40):
    # your code here
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    return logits


def top_p_filtering(logits, top_p=0.9):
    # your code here
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits


def train(model, dataset, lr, out_seq_len, num_epochs):
    model = model.to(device)
    optimizer = model.get_optimizer(lr)
    loss_function = model.get_loss_function()
    avg_loss = []
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0
        for in_seq, out_seq in dataset.get_example():
            in_seq, out_seq = in_seq.to(device), out_seq.to(device)

            optimizer.zero_grad()
            if isinstance(model, CharLSTM):
                out, _, _ = model(in_seq)
            else:
                out, _ = model(in_seq)
            loss = loss_function(out, out_seq)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}. Average loss: {(running_loss / dataset.examples_per_epoch):.8f}")
        avg_loss.append(running_loss / dataset.examples_per_epoch)

    print(f"Training time: {time.time() - start_time:.2f}s")

    # Plot the loss
    import matplotlib.pyplot as plt
    plt.plot(avg_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss')
    plt.savefig('training_loss.png', dpi=300)
    plt.show()

    # Sample from the model and print the result
    with torch.no_grad():
        for i in range(5):  # Include 5 fix examples
            # Test different temperatures
            starting_char = dataset.mappings['char_to_idx'][dataset.unique_chars[i]]
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.1)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {0.1}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {0.5}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=1.0)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {1.0}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=3.0)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {3.0}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=5.0)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {5.0}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=10.0)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"Temp: {10.0}", ''.join(generated_seq))

            # Different top-k values
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_k=1)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-k: {1}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_k=5)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-k: {5}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_k=10)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-k: {10}", ''.join(generated_seq))

            # Different top-p values
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_p=0.1)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-p: {0.1}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_p=0.5)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-p: {0.5}", ''.join(generated_seq))
            generated_seq = model.sample_sequence(starting_char, out_seq_len, temp=0.5, top_p=0.9)
            generated_seq = dataset.convert_indices_to_seq(generated_seq)
            print(f"TOP-p: {0.9}", ''.join(generated_seq))

    return model


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 200
    epoch_size = 64  # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, epoch_size)
    model = CharRNN(dataset.vocab_size, embedding_size, hidden_size)
    train(model, dataset, lr=lr,
          out_seq_len=out_seq_len,
          num_epochs=num_epochs)


def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 200
    epoch_size = 64
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    dataset = CharSeqDataloader(data_path, seq_len, epoch_size)
    model = CharLSTM(dataset.vocab_size, embedding_size, hidden_size)
    train(model, dataset, lr=lr,
          out_seq_len=out_seq_len,
          num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):
    # Convert lists to tensors and reverse sequences for the backward LSTM
    batch_premises_tensors = [torch.tensor(p) for p in batch_premises]
    batch_hypotheses_tensors = [torch.tensor(h) for h in batch_hypotheses]

    batch_premises_reversed = [torch.tensor(p[::-1]) for p in batch_premises]
    batch_hypotheses_reversed = [torch.tensor(h[::-1]) for h in batch_hypotheses]

    # Pad sequences
    batch_premises_padded = pad_sequence(batch_premises_tensors, batch_first=True)
    batch_hypotheses_padded = pad_sequence(batch_hypotheses_tensors, batch_first=True)

    batch_premises_reversed_padded = pad_sequence(batch_premises_reversed, batch_first=True)
    batch_hypotheses_reversed_padded = pad_sequence(batch_hypotheses_reversed, batch_first=True)

    return batch_premises_padded, batch_hypotheses_padded, batch_premises_reversed_padded, batch_hypotheses_reversed_padded


def create_embedding_matrix(index_map, emb_dict, emb_dim):
    # Initialize matrix with zeros of type float32
    embedding_matrix = np.zeros((len(index_map), emb_dim), dtype=np.float32)

    # Loop through the word_index dictionary
    for word, i in index_map.items():
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector.astype(np.float32)  # Ensure the vector is float32

    return torch.from_numpy(embedding_matrix)


def evaluate(model, dataloader, index_map):
    # Ensure the model is in evaluation mode so that layers like dropout are disabled
    model.eval()

    # Keep track of the number of correct predictions and the total number of predictions
    correct_predictions = 0
    total_predictions = 0

    # Disable gradient calculations for efficiency and to prevent changes during evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch data
            premises, hypotheses, labels = batch["premise"], batch["hypothesis"], batch["label"]
            # print(batch)
            # print(premises, hypotheses, labels)

            premises = tokenize(premises)
            hypotheses = tokenize(hypotheses)

            # Convert premises and hypotheses to tensors of word indices
            premises_indices = tokens_to_ix(premises, index_map)
            hypotheses_indices = tokens_to_ix(hypotheses, index_map)

            # Pad the sequences so that each sequence in the batch has the same length
            # premises_padded = pad_sequence([torch.tensor(seq) for seq in premises_indices], batch_first=True).to(device)
            # hypotheses_padded = pad_sequence([torch.tensor(seq) for seq in hypotheses_indices], batch_first=True).to(
            #     device)
            # Convert labels to tensor
            # label_to_index = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

            # labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long).to(device)

            # Make predictions using the model
            outputs = model(premises_indices, hypotheses_indices)

            # Choose the predicted class with the highest score for each input
            _, predicted = torch.max(outputs, 1)

            # Update the correct predictions and total predictions counters
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate the accuracy as the number of correct predictions divided by the total number of predictions
    print(correct_predictions, total_predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # Initialize the layers as per the architecture
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # The intermediate layer takes the concatenated cell states so its input size is hidden_dim * 2
        self.int_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        premises_padded, hypotheses_padded, _, _ = fix_padding(a, b)  # Pad and reverse the sequences
        # Move to device
        premises_padded = premises_padded.to(device)
        hypotheses_padded = hypotheses_padded.to(device)
        # Embed the sequences
        embedded_premises = self.embedding_layer(premises_padded)  # (batch_size, seq_len, hidden_dim)
        embedded_hypotheses = self.embedding_layer(hypotheses_padded)  # (batch_size, seq_len, hidden_dim)

        # Process sequences through the LSTM
        _, (_, premises_final_cell_state) = self.lstm(embedded_premises)  # (batch_size, seq_len, hidden_dim)
        _, (_, hypotheses_final_cell_state) = self.lstm(embedded_hypotheses)  # (batch_size, seq_len, hidden_dim)

        # Concatenate the final cell states of the premises and hypotheses
        concatenated_final_states = torch.cat((premises_final_cell_state[-1], hypotheses_final_cell_state[-1]),
                                              dim=1)  # (batch_size, hidden_dim * 2)

        # Pass the concatenated final states through the linear layers
        intermediate_output = F.relu(self.int_layer(concatenated_final_states))
        output = self.out_layer(intermediate_output)

        return output


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # your code here
        # Initialize the layers as per the architecture
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm_forward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # The intermediate layer takes the concatenated cell states from both LSTMs so its input size is hidden_dim * 4
        self.int_layer = nn.Linear(hidden_dim * 4, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        # Pad and reverse the sequences
        premises_padded, hypotheses_padded, premises_padded_reversed, hypotheses_padded_reversed = fix_padding(a,
                                                                                                               b)

        # Embed the sequences
        embedded_premises = self.embedding_layer(premises_padded)
        embedded_hypotheses = self.embedding_layer(hypotheses_padded)
        embedded_premises_reversed = self.embedding_layer(premises_padded_reversed)
        embedded_hypotheses_reversed = self.embedding_layer(hypotheses_padded_reversed)

        # Process sequences through the LSTMs
        _, (_, premises_final_cell_state_forward) = self.lstm_forward(embedded_premises)
        _, (_, hypotheses_final_cell_state_forward) = self.lstm_forward(embedded_hypotheses)
        _, (_, premises_final_cell_state_backward) = self.lstm_backward(embedded_premises_reversed)
        _, (_, hypotheses_final_cell_state_backward) = self.lstm_backward(embedded_hypotheses_reversed)

        # Concatenate the final cell states from both the forward and backward LSTMs
        concatenated_final_states = torch.cat(
            (premises_final_cell_state_forward[-1], premises_final_cell_state_backward[-1],
             hypotheses_final_cell_state_forward[-1], hypotheses_final_cell_state_backward[-1]), dim=1)

        # Pass the concatenated final states through the linear layers
        intermediate_output = F.relu(self.int_layer(concatenated_final_states))
        output = self.out_layer(intermediate_output)

        return output


def run_snli(model):
    dataset = load_datasets("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = ""  # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered = dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders
    dataloader_train = DataLoader(train_filtered, batch_size=32, shuffle=True)
    dataloader_valid = DataLoader(valid_filtered, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(test_filtered, batch_size=32, shuffle=True)
    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code


def run_snli_lstm():
    model_class = ""  # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)


def run_snli_bilstm():
    model_class = ""  # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)


if __name__ == '__main__':
    # run_char_rnn()
    run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
