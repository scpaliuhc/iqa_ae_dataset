import torch

class NES(object):
    def __init__(self,model,steps,eps,alpha,device):
        self.model=model
        self.steps=steps
        self.eps=eps
        self.device=device
        self.alpha=alpha

    def nes_grad_est(self,x, y, sigma=1e-3, n = 10):
        g = torch.zeros(x.size()).to(self.device)
        g = g.view(x.size()[0],-1)
        y = y.view(-1,1)
        for _ in range(n):
            u = torch.randn(x.size()).to(self.device)
            out1 = self.model(x+sigma*u)
            out2 = self.model(x-sigma*u)
            out1 = torch.gather(out1,1,y)
            #pdb.set_trace()
            out2 = torch.gather(out2,1,y)
            #print(out1.size(),u.size(),u.view(x.size()[0],-1).size())
            #print(out1[0][y],out2[0][y])
            g +=  out1 * u.view(x.size()[0],-1)
            g -=  out2 * u.view(x.size()[0],-1)
        g=g.view(x.size())
        return -1/(2*sigma*n) * g


    def nes(self, x_in, y, TARGETED):
        if self.eps == 0:
            return x_in
        x_adv = x_in.clone()
        # lr = 0.01
        for i in range(self.steps):
            #print(f'\trunning step {i+1}/{steps} ...')
            # print(net.predict(x_adv)[0][y].item())
            if TARGETED:
                step_adv = x_adv - self.alpha * torch.sign(self.nes_grad_est(x_adv, y))
            else:
                step_adv = x_adv + self.alpha * torch.sign(self.nes_grad_est(x_adv, y))
            diff = step_adv - x_in
            diff.clamp_(-self.eps, self.eps)
            x_adv = x_in + diff
            x_adv.clamp_(0.0, 1.0)
            if not TARGETED and i>0 and i%5==0:
                pred=self.model(x_adv)
                pred=torch.argmax(pred,keepdim=True).view([-1])
                if pred[0]!=y[0]:
                    break
        return x_adv

    def __call__(self, input_xi, label_or_target, TARGETED=False):
        with torch.no_grad():
            return self.nes(input_xi,label_or_target,TARGETED=TARGETED)