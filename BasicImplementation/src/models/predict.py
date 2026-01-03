from train import train

def predict(xTest):

    model = train()
    predY = model.predict(xTest)
    return predY