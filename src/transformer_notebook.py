import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import os
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---Load raw data
data = pd.read_csv('raw data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv')

# ---Prepare data
data.columns = data.columns.str.lower().str.replace(" ", "_")
data['main_category'] = data['category'].str.split("|").str[0].str.strip()
data.loc[data['main_category']=='Remote & App Controlled Vehicles & Parts', 'main_category'] = 'Remote & App Controlled Vehicle Parts' # fix value discrepency
data['target_text'] = data[['product_name','about_product', 'product_specification', 'technical_details']].apply(
    lambda row: " ".join(str(value) if pd.notnull(value) else "" for value in row),
    axis=1
)
label_transform = LabelEncoder().fit_transform(data['main_category'])
data['label']=label_transform
# data[['label','main_category']].value_counts()

# ---Fine-tune sentence transformer model
# Choose the sentence transformer model for semantic searching
model_name = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Define classification head for categorization.
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    def forward(self, features):
        x = features['sentence_embedding']
        x = self.linear(x)
        return x # Output raw logits

num_classes = 24
classification_head = ClassificationHead(model.get_sentence_embedding_dimension(), num_classes)

# Combine SentenceTransformer model and classification head.
class SentenceTransformerWithHead(nn.Module):
    def __init__(self, transformer, head):
        super(SentenceTransformerWithHead, self).__init__()
        self.transformer = transformer
        self.head = head

    def forward(self, input):
        features = self.transformer(input)
        logits = self.head(features)
        return logits

model_with_head = SentenceTransformerWithHead(model, classification_head)

# Tune model
train_sentences = data['target_text']
train_labels = data['label']
# training parameters
num_epochs = 5
batch_size = 2
learning_rate = 2e-5

# Convert the dataset to PyTorch tensors.
train_examples = [InputExample(texts=[s], label=l) for s, l in zip(train_sentences, train_labels)]

# Customize collate_fn to convert InputExample objects into tensors.
def collate_fn(batch):
    texts = [example.texts[0] for example in batch]
    labels = torch.tensor([example.label for example in batch])
    return texts, labels


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

# Define the loss function, optimizer, and learning rate scheduler.
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model_with_head.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
loss_list = []
for epoch in range(num_epochs):
    model_with_head.train()
    for step, (texts, labels) in enumerate(train_dataloader):
        labels = labels.to(model.device)
        optimizer.zero_grad()

        # Encode text and pass through classification head.
        inputs = model.tokenize(texts)
        input_ids = inputs['input_ids'].to(model.device)
        input_attention_mask = inputs['attention_mask'].to(model.device)
        inputs_final = {'input_ids': input_ids, 'attention_mask': input_attention_mask}

        # move model_with_head to the same device
        model_with_head = model_with_head.to(model.device)
        logits = model_with_head(inputs_final)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    model_save_path = f'./intermediate-output/epoch-{epoch}'
    model.save(model_save_path)
    loss_list.append(loss.item())
# Save the final model
model_final_save_path = 'st_ft_epoch_5'
model.save(model_final_save_path)



