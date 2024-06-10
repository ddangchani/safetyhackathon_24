# Prediction function for STGCN model

def predict_stgcn(model, data, adj_mx, n_his, n_pred, scaler, device, batch_size):
    """
    Function to predict the output of the STGCN model.
    
    Arguments:
    model -- STGCN model
    data -- data to be predicted
    adj_mx -- adjacency matrix
    n_his -- number of historical time steps
    n_pred -- number of future time steps
    scaler -- scaler object
    device -- device to run the model
    batch_size -- batch size
    
    Returns:
    y_pred -- predicted output
    """
    data_col = data.shape[0]
    x, y = data_transform(data, n_his, n_pred, device)
    data = TensorDataset(x, y)
    data_iter = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            y_pred.append(scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1))
    
    return np.array(y_pred).reshape(-1)
