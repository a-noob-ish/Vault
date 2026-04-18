import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------
# VAULT OF THE LUCII
# Custom AI Engineering Tools & Auto-ML Pipelines
# ---------------------------------------------------------

# Paste your pair_data_to_trained_model function below this line!
def pair_data_to_trained_model(list_of_pair_data,hidden_layer_size):
    
    train_data=list_of_pair_data[0:int(len(list_of_pair_data)*0.7)]
    test_data=list_of_pair_data[int(len(list_of_pair_data)*0.7):int(len(list_of_pair_data)*0.9)]
    acc_measurement_data=list_of_pair_data[int(len(list_of_pair_data)*0.9): ]

    sample_image, sample_label = list_of_pair_data[0]
    input_size = sample_image.shape[1] * sample_image.shape[2]
    output_size=len(torch.unique(torch.tensor([b for a,b in list_of_pair_data])))
    
    train_batched=DataLoader(train_data, batch_size=int(len(train_data)/100), shuffle=True)
    test_batched=DataLoader(test_data, batch_size=int(len(list_of_pair_data)/100), shuffle=False)
    
    layer1=torch.nn.Linear(in_features=input_size, out_features=hidden_layer_size, bias=True)
    layer2=torch.nn.Linear(in_features=hidden_layer_size, out_features=output_size,bias=True)
    
    flow=nn.Sequential(nn.Flatten(),layer1,nn.ReLU(),layer2)  # input to output  flow, just feed it data to get values
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(flow.parameters(), lr=0.1)
    
    i=0
    while True:
        cumul_loss=0.0
        for batch_images,batch_label in train_batched:
            loss=loss_fn(flow(batch_images),batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumul_loss += loss.item()*len(batch_images) 
        global_train_loss = cumul_loss/len(train_data)
        i=i+1
        
        with torch.no_grad():
            total_test_loss = 0.0
            for test_images, test_labels in test_batched:
                loss = loss_fn(flow(test_images), test_labels)
                total_test_loss += loss.item() * len(test_images)
            global_test_loss = total_test_loss / len(test_data)

        print(f"step {i} ---> global training loss = {global_train_loss} and global test loss = {global_test_loss}")
        
        if global_train_loss < 0.33 and global_test_loss < 0.33:
            break

    return flow

### stored in "vault_of_the_lucii.py"    
### This function takes in list of pairs (thing,category/label) and spits out a model trained on that list that predict the probability distribution
###      for any similar thing to be in any category/label. No prior knowledge of training pair data needed. But, each data must match in size. 
### Pros: simple concept, generalizes for other very different (thing,category)-classification. Checked for thing=image but generalizes for 
###      other things too. 
### Cons: 1) inputdata must be a list of (2D matrix,label)-pair and all matrix must be same size. 2) It will crash memory if input datasize is large.
###       1 may always be arranged without much sacrifice, I guess and 2 can be fixed with bit modified code of the function. 









































































