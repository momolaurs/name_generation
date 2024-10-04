"""
NLP Assignment 2: Character-based LM for name generation - training

Authors: Ángela María Gómez Zuluaga and Laura Moset Estruch

Original file is located at
    https://colab.research.google.com/drive/1Hw-LXJ3YfjBRoTk_0gGy-_QTDORufTVw
"""

# Commented out IPython magic to ensure Python compatibility.
import torch #to make the model
import torch.nn.functional as F
import pandas as pd #reading the xls file with the modern names
import matplotlib.pyplot as plt #creating the loss plot
# %matplotlib inline

import gdown

id_m = "1_x4_o961lfxQ46wHHN0YcBNH8ZURI_IO" #modern: nombres_por_edad_media.xls
output_m = "nombres_por_edad_media.xls"
gdown.download(id=id_m, output=output_m)

id_o = "1--wzxkim29aBgqy3WZHLnDcWdkzKXcfs" #older: 0_old_names_final.txt
output_o ="0_old_names_final.txt"
gdown.download(id=id_o, output=output_o)

#Function for obtaining names from excel sheet and converting to list
def get_names(sheet_name):
  df = pd.read_excel('nombres_por_edad_media.xls', sheet_name=sheet_name, skiprows=6) #Sheet name because there's a tab for each sex + skip the first 6 rows with text
  filtered_names = df[df['Edad Media (*)'] <= 60]['Nombre'].tolist() #We only want the names with an average age lower than 60ys
  filtered_names = filtered_names[:500] #We only keep the first 500 names
  return [str(name).lower() for name in filtered_names]

male_names = get_names('Hombres')
female_names = get_names('Mujeres')

#Combine into a single list
#modern dataset
names_modern = male_names + female_names
print("\n\nModern dataset---")
print("Len: ", len(names_modern))
print("Sample:\n")
print(names_modern[:10]) #Confirm correct formatting

#old dataset
names_old = open('0_old_names_final.txt', 'r').read().splitlines()
names_old = [name.lower() for name in names_old]
print("\n\nOld dataset---")
print("Len: ", len(names_old))
print("Sample:\n")
print(names_old[:10]) #Confirm correct formatting

#########################################
    ###### MODERN NAMES MODEL ######
#########################################

def name_generator(a_list, ds_name):

    """
    Trains a character-based language model for name generation

    Parameters:
    - a_list (list): dataset of names used for training the model in list format
    - ds_name (str): name or identifier for the dataset being used in string format.
    """

  separation = "\n" + "-" * 40 + "\n" #used along the function to better visualize the output
  # build the vocabulary of characters and mappings to/from integers
  chars = sorted(list(set(''.join(a_list))))
  stoi = {s:i+1 for i,s in enumerate(chars)}
  stoi['.'] = 0
  itos = {i:s for s,i in stoi.items()}
  print(f"Character Vocabulary and Mapping: {itos}")

  print(separation)

  # build the dataset
  block_size = 3 #sets context length: how many characters do we take to predict the next one?
  def build_dataset(words):
    X, Y = [], [] #The 'X' are the input to the neural net, and the 'Y' are the labels for each example inside X

    for w in words:
      #print(w)
      context = [0] * block_size #padded context of 0 tokens
      for ch in w + '.': #iterate over all characters
        ix = stoi[ch] #get the character in the sequence
        X.append(context) # build out the array 'X' which stores the current context
        Y.append(ix) #build out array 'Y' of the current character
        context = context[1:] + [ix] #crop the context and append the new character in the sequence

    X = torch.tensor(X)
    Y = torch.tensor(Y) #this is the next character in the sequence we want to predict
    print(X.shape, Y.shape)
    return X, Y

  #### split ####
  import random
  random.seed(23)
  random.shuffle(a_list)
  n1 = int(0.8*len(a_list)) #80% of words
  n2 = int(0.9*len(a_list)) #90% of words

  Xtr, Ytr = build_dataset(a_list[:n1]) #indexing up to n1 (80%) for TRAINING
  Xdev, Ydev = build_dataset(a_list[n1:n2]) #indexing between n1 and n2 (80%-90%) for VALIDATION
  Xte, Yte = build_dataset(a_list[n2:]) #indexing last 10% percent for TESTING

  #########beginning of hyperparameters#############

  g = torch.Generator().manual_seed(2147483647) #torch generator with manual seed for reproducibility

  #matrix of embeddings -> lookup table. The integers index into the lookup table. First layer of NN.
  C = torch.randn((29, 10), generator=g) #29 rows (number of chrs) in a 10-dimensional space (10 columns). Each of the 29 chrs will have a 10-dimensional embedding

  #hidden layer
  W1 = torch.randn((30, 100), generator=g) #the 30 matches the shape of the 'emb' (3 for the block size * 10 for the embedding size)
  b1 = torch.randn(100, generator=g)

  #softmax layer
  W2 = torch.randn((100, 29), generator=g)
  b2 = torch.randn(29, generator=g)

  #clustered parameters
  parameters = [C, W1, b1, W2, b2]

  ###########end of hyperparameters################


  for p in parameters: #setting the gradient requirement
    p.requires_grad = True

  #initializing lists for tracking metrics
  lossi = []
  stepi = []

  print(separation)

  for i in range(150000): #number of steps
  # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

  #### forward pass ####
    emb = C[Xtr[ix]]

    #hidden layer of activations
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # By using -1, we are allowing pytorch to infer the real size by seeing the other value, instead of hardcoding it
    logits = h @ W2 + b2 #defining logits: raw output of NN
    loss = F.cross_entropy(logits, Ytr[ix]) #calculating loss
    #print(loss.item())

  #### backward pass ####
    for p in parameters:
      p.grad = None #set the gradients to 0
    loss.backward() #populate the gradients

    # update
    lr = 0.7 if i < 75000 else 0.05 #step decay
    for p in parameters:
      p.data += -lr * p.grad #adjusting parameters based on gradient descent

    # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())

  #Creating the plot
  print("Downloading loss plot...")
  plt.plot(stepi, lossi)
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.title('Training Loss Over Steps')
  file_name = ds_name + "_training_loss_plot.png"
  plt.savefig(file_name)
  plt.close()
  print(f"Done! Look out for {file_name}")

  print(separation)

  # training loss
  emb = C[Xtr] #Get embeddings
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # Apply tanh activation to hidden layer
  logits = h @ W2 + b2  # Calculate raw output (logits) of the neural network
  loss = F.cross_entropy(logits, Ytr) # Calculate the loss using cross-entropy
  print(f"Training loss: {loss}")

  # validation loss
  emb = C[Xdev]
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ydev)
  print(f"Validation loss: {loss}")

  # test loss
  emb = C[Xte]
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Yte)
  print(f"Test loss: {loss}")

  print(separation)

  # generation sample
  print("Model's Generation Attempt (50 names):")
  generated_names = []
  g = torch.Generator().manual_seed(2147483647)
  for _ in range(50): #number of samples
    out = []
    context = [0] * block_size #initialize with all "..."
    while True:
      emb = C[torch.tensor([context])] #embed context using embedding table "C"
      h = torch.tanh(emb.view(1, -1) @ W1 + b1) #project into hidden space
      logits = h @ W2 + b2 # Calculate raw output (logits) of the neural network
      probs = F.softmax(logits, dim=1) #calculate probabilities. Exponentiates the logits and makes them sum to 1, which avoid overflows.
      ix = torch.multinomial(probs, num_samples=1, generator=g).item() #get next index
      context = context[1:] + [ix] #shift context window to append the index
      out.append(ix)
      if ix == 0: #until we generate the 0th character again
        break

    generated_name = ''.join(itos[i] for i in out)
    generated_names.append(generated_name)
    print(generated_name)

  return generated_names

print("\n\nTraining the model on the Modern Dataset:\n")
model_modern = name_generator(names_modern, "Modern_ds")

print("\n\nTraining the model on the 1880-1940 Dataset:\n")
model_older = name_generator(names_old, "1880-1940_ds")

def names_in_ds(generated, dataset): #function to check how many of the generated names were in the og dataset
  in_ds = []
  no_period = [name.replace('.', '') for name in generated]
  for item in no_period:
    if item in dataset:
      in_ds.append(item)
    else:
      pass
  print(f"{len(in_ds)} out of {len(generated)} generated names were in the original dataset")
  print(f"Names in the original dataset: {in_ds}")
  return "Nice!"

print("\n\nModern Dataset:")
print(names_in_ds(model_modern, names_modern))

print("\n\n1880-1940 Dataset:")
print(names_in_ds(model_older, names_old))

# References:
# [1] Karpathy, A. (2022). makemore. Retrieved from https://github.com/karpathy/makemore. Last accessed on March 1, 2024.
# [2] Instituto Nacional de Estadística. (2023). Todos los nombres con frecuencia igual o mayor a 20 personas. Last accessed on March 1, 2024. https://www.ine.es/uc/nDER2igi
# [3] José Calvo Tello (Comp.) (2017). Corpus de novelas de la Edad de Plata Würzburg: CLiGS, 2017. Last accessed on February 29, 2024. https://github.com/cligs/textbox/tree/master/spanish/novela-espanola
