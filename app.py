from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow
import torchvision.models as models
import math
from torchvision import transforms, datasets, models


app = Flask(__name__)


"""################################# MODEL LOADING from CHECKPOINT ##############################################"""

# Basic details

path = 'resnet50-transfer.pth'
# Get the model name
model_name = path.split('-')[0]

checkpoint = torch.load(path, map_location = torch.device('cpu'))


if model_name == 'resnet50':
    model = models.resnet50( weights = models.ResNet50_Weights.DEFAULT)
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
    model.fc = checkpoint['fc']

# Load in the state dict
model.load_state_dict(checkpoint['state_dict'])

# Model basics
model.class_to_idx = checkpoint['class_to_idx']
model.idx_to_class = checkpoint['idx_to_class']
model.epochs = checkpoint['epochs']



model.class_to_idx = [('Alpinia Galanga (Rasna)', 0),
 ('Amaranthus Viridis (Arive-Dantu)', 1),
 ('Artocarpus Heterophyllus (Jackfruit)', 2),
 ('Azadirachta Indica (Neem)', 3),
 ('Basella Alba (Basale)', 4),
 ('Brassica Juncea (Indian Mustard)', 5),
 ('Carissa Carandas (Karanda)', 6),
 ('Citrus Limon (Lemon)', 7),
 ('Ficus Auriculata (Roxburgh fig)', 8),
 ('Ficus Religiosa (Peepal Tree)', 9),
 ('Hibiscus Rosa-sinensis', 10),
 ('Jasminum (Jasmine)', 11),
 ('Mangifera Indica (Mango)', 12),
 ('Mentha (Mint)', 13),
 ('Moringa Oleifera (Drumstick)', 14),
 ('Muntingia Calabura (Jamaica Cherry-Gasagase)', 15),
 ('Murraya Koenigii (Curry)', 16),
 ('Nerium Oleander (Oleander)', 17),
 ('Nyctanthes Arbor-tristis (Parijata)', 18),
 ('Ocimum Tenuiflorum (Tulsi)', 19),
 ('Piper Betle (Betel)', 20),
 ('Plectranthus Amboinicus (Mexican Mint)', 21),
 ('Pongamia Pinnata (Indian Beech)', 22),
 ('Psidium Guajava (Guava)', 23),
 ('Punica Granatum (Pomegranate)', 24),
 ('Santalum Album (Sandalwood)', 25),
 ('Syzygium Cumini (Jamun)', 26),
 ('Syzygium Jambos (Rose Apple)', 27),
 ('Tabernaemontana Divaricata (Crape Jasmine)', 28),
 ('Trigonella Foenum-graecum (Fenugreek)', 29)]

model.idx_to_class = [(0, 'Alpinia Galanga (Rasna)'),
 (1, 'Amaranthus Viridis (Arive-Dantu)'),
 (2, 'Artocarpus Heterophyllus (Jackfruit)'),
 (3, 'Azadirachta Indica (Neem)'),
 (4, 'Basella Alba (Basale)'),
 (5, 'Brassica Juncea (Indian Mustard)'),
 (6, 'Carissa Carandas (Karanda)'),
 (7, 'Citrus Limon (Lemon)'),
 (8, 'Ficus Auriculata (Roxburgh fig)'),
 (9, 'Ficus Religiosa (Peepal Tree)'),
 (10, 'Hibiscus Rosa-sinensis'),
 (11, 'Jasminum (Jasmine)'),
 (12, 'Mangifera Indica (Mango)'),
 (13, 'Mentha (Mint)'),
 (14, 'Moringa Oleifera (Drumstick)'),
 (15, 'Muntingia Calabura (Jamaica Cherry-Gasagase)'),
 (16, 'Murraya Koenigii (Curry)'),
 (17, 'Nerium Oleander (Oleander)'),
 (18, 'Nyctanthes Arbor-tristis (Parijata)'),
 (19, 'Ocimum Tenuiflorum (Tulsi)'),
 (20, 'Piper Betle (Betel)'),
 (21, 'Plectranthus Amboinicus (Mexican Mint)'),
 (22, 'Pongamia Pinnata (Indian Beech)'),
 (23, 'Psidium Guajava (Guava)'),
 (24, 'Punica Granatum (Pomegranate)'),
 (25, 'Santalum Album (Sandalwood)'),
 (26, 'Syzygium Cumini (Jamun)'),
 (27, 'Syzygium Jambos (Rose Apple)'),
 (28, 'Tabernaemontana Divaricata (Crape Jasmine)'),
 (29, 'Trigonella Foenum-graecum (Fenugreek)')]

"""################################## PRE PROCESS THE IMAGE #####################################################"""


# Transform the Image
def image_transform(image_path):
    t = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = t(image)
    return image



def process_image(image_path):
    """Process an image path into a PyTorch tensor"""
    image = image_transform(image_path)  #********#
    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256
    img = img[:3,:,:]

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    print(img.shape, means.shape)
    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor




"""######################################  PREDICTION FUNCTION  ################################################"""

def predict(image_path, model, topk ):

    """
    Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return
    --------
    Returns
    """

    img_tensor = process_image(image_path)

    img_tensor = img_tensor.reshape(1, 3, 224, 224)

    with torch.no_grad():
        # Set to evaluation
        model.eval()

        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(topk, dim = 1)

        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes




"""##########################################   ROUTES    ##################################################"""

# Home Route
@app.route('/')
def index():
    return render_template("index.html")

# when the user hits submit button
@app.route('/submit', methods = ['GET', 'POST'])
@app.route('/submit/', methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files["my_image"]

        img_path = "static/" + img.filename
        img.save(img_path)
        img = image_transform(img_path)

        # Predict Function, takes (imagePath, modelName, number of top precitions to return) as parameters
        img, p, classes = predict(img_path, model, 1)
        result = pd.DataFrame({'p': p}, index = classes)

        p = result.sort_values('p')['p']

    return render_template("index.html", prediction = p, img_path = img_path)


"""##################################### MAIN APP CALL #########################################"""
if __name__ == "__main__":
    app.run( debug = True )

