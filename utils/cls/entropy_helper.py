import torch
import torch.nn.functional as F


# Helper for Entropy 
def get_entropy(model, dataloader):
    model.eval()
    entropy_list = []
    
    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()
            output = model(image)
            logit = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logit, dim=1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=1)
            entropy_list.append(entropy)
            
    return torch.cat(entropy_list)