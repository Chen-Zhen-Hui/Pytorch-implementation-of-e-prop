import torch.nn as nn
import torch.nn.functional as F
from config import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

def p(x):
    print(x.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_b=1000  # batch size
n_in=784 # 输入层
n_rec=256 # 隐藏层层
n_out=10 # 输出层
n_t = 10 # 时间窗口
w_init_gain=[0.5,0.1,0.5]
alpha=0.8 # LIF的衰减系数
kappa=0.8
gamma=1.0
thr=1.0
classif=True # 分类任务
# timestep=10
class SRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters
        self.w_in = nn.Parameter(torch.Tensor(n_rec, n_in))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.reset_parameters(w_init_gain)

    def reset_parameters(self, gain):
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data
        # Weight gradients

    def init_net(self):
        self.v = torch.zeros(n_b, n_rec).to(device)
        self.vo = torch.zeros(n_b, n_out).to(device)
        self.z = torch.zeros(n_b, n_rec).to(device)

        self.Fkz = torch.zeros(n_b, n_rec).to(device)  # kappa filter for z
        self.Faz = torch.zeros(n_b, n_rec).to(device)  # alpha filter for z
        self.Fax = torch.zeros(n_b, n_in).to(device)  # alpha filter for x
        self.L = torch.zeros(n_b, n_rec).to(device)  # Learning signal
        self.e_rec = torch.zeros(n_b, n_rec, n_rec).to(device)  # eligibility for w_rec
        self.e_in = torch.zeros(n_b, n_rec, n_in).to(device)  # eligibility for w_in
        self.h = torch.zeros(n_b, n_rec).to(device)  # pseudo
        self.Fke_rec = torch.zeros(n_b, n_rec, n_rec).to(device)  # kappa filter for e_rec
        self.Fke_in = torch.zeros(n_b, n_rec, n_in).to(device)  # kappa filter for e_in

        self.w_in.grad = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)

        
    def forward(self, x, do_training, yt):
        self.w_rec *= (1 - torch.eye(n_rec, n_rec, device=device))  # Cancel self connected neurons
        self.v = alpha * self.v * (1 - self.z) + torch.mm(self.z, self.w_rec.t()) + torch.mm(x, self.w_in.t())
        self.Faz = self.Faz * alpha + self.z  # n_b, n_r
        self.z = (self.v > thr).float()
        self.vo = kappa * self.vo + torch.mm(self.z, self.w_out.t())
        if classif:  # Apply a softmax function for classification problems
            yo = F.softmax(self.vo, dim=-1)
        else:
            yo = self.vo
        if do_training:
            self.h = gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - thr) / thr))  # n_b, n_r

            self.L = torch.mm((yo - yt), self.w_out)  # n_b, n_r

            self.Fkz = self.Fkz * kappa + self.z  # n_b, n_r
            self.w_out.grad += torch.mm((yo - yt).t(), self.Fkz)

            self.e_rec = self.h.unsqueeze(2) * self.Faz.unsqueeze(1)
            self.Fke_rec = self.Fke_rec * kappa + self.e_rec  # n_b, n_r, n_r
            self.w_rec.grad += torch.sum(self.L.unsqueeze(2) * self.Fke_rec, dim=0)

            self.Fax = self.Fax * alpha + x  # n_b, n_i
            self.e_in = self.h.unsqueeze(2) * self.Fax.unsqueeze(1)
            self.Fke_in = self.Fke_in * kappa + self.e_in  # n_b, n_i, n_r
            self.w_in.grad += torch.sum(self.L.unsqueeze(2) * self.Fke_in, dim=0)
        return yo
    

if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])
    root='mnist/' # 自己的路径
    train_dataset = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=n_b, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=n_b, shuffle=False)

    model=SRNN()
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.000003)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for epoch in range(100):
            running_loss = 0.0
            total=0
            correct=0
            model.train()
            for data, label in train_loader:
                data, label = data.to(device).view(n_b,-1), label.to(device)
                label_onehot = F.one_hot(label, num_classes=n_out)
                out = torch.zeros(n_t,n_b,n_out, device=device)
                model.init_net()
                for t in range(n_t):
                    out[t] = model(data, True, label_onehot)

                # out = out.mean(dim=0)
                out = out[-1]
                loss = criterion(out, label)
                running_loss += loss.item()
                _, predicted = torch.max(out, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{100}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

            test_loss = 0.0
            total=0
            correct=0
            model.eval()
            for data, label in test_loader:
                
                data, label = data.to(device).view(n_b,-1), label.to(device)
                label_onehot = F.one_hot(label, num_classes=n_out)
                out = torch.zeros(n_t,n_b,n_out, device=device)
                for t in range(n_t):
                    out[t] = model(data, False, label_onehot)

                # out = out.mean(dim=0)
                out = out[-1]
                test_loss += criterion(out, label).item()
                _, predicted = torch.max(out, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            print(f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
                
            
