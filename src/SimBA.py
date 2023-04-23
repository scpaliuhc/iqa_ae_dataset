import torch
import torch.nn.functional as F
# import utils


class SimBA:
    
    def __init__(self,model,device,steps=10000, eps=0.2, targeted=False):
        self.model = model
        # self.dataset = dataset
        # self.model.eval()
        self.device=device
        self.num_iters=steps
        self.epsilon=eps
        self.targeted=targeted

        
    # def normalize(self, x):
    #     return utils.apply_normalization(x, self.dataset)

    def get_probs(self, x, y):
        output = self.model(x)
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    # def get_preds(self, x):
    #     output = self.model(self.normalize(x.cuda())).cpu()
    #     _, preds = output.data.max(1)
    #     return preds

    # 20-line implementation of SimBA for single image input
    def __call__(self, x, y, ):
        y=y[0]
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        # x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(self.num_iters):
            diff = torch.zeros(n_dims).to(self.device)
            diff[perm[i]] = self.epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if self.targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if self.targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if not self.targeted and i>0 and i%20==0:
                pred=self.model(x)
                pred=torch.argmax(pred,keepdim=True).view([-1])
                if pred[0]!=y:
                    break
        return x.detach()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    