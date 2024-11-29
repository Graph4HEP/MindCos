import mindspore as ms
import mindspore.nn as nn
from mindspore import context

from src.dataset import retrieve_dataloaders as loader
from src.model import LorentzNet
from src.train import train_loop, test_loop, forward_fn

context.set_context(mode=1, device_target="CPU")

dataset, dataloaders = loader(32, 100000)

model = LorentzNet(n_scalar = 8, n_hidden = 72, n_class = 2,
                       dropout = 0.2, n_layers = 6,
                       c_weight = 0.001)

lr = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0008535533905932737, 0.0005, 0.00014644660940672628, 0.001, 0.0009619397662556434, 0.0008535533905932737, 0.000691341716182545, 0.0005, 0.0003086582838174551, 0.00014644660940672628, 3.806023374435663e-05, 0.001, 0.0009903926402016153, 0.0009619397662556434, 0.0009157348061512727, 0.0008535533905932737, 0.0007777851165098011, 0.000691341716182545, 0.0005975451610080642, 0.0005, 0.00040245483899193594, 0.0003086582838174551, 0.00022221488349019903, 0.00014644660940672628, 8.426519384872733e-05, 3.806023374435663e-05, 9.607359798384786e-06, 4.803679899192393e-06, 2.4018399495961964e-06, 1.2009199747980982e-06]
Nbatch = len(dataloaders['train']) 
lr = [x for x in lr for _ in range(Nbatch)]

optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
print('Train')
for t in range(35):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, dataloaders['train'], loss_fn, grad_fn, optimizer, t, 'LorenzNet_training_CPU_35ep_papar_lr')
    print()
    test_loop(model, dataloaders['val'], loss_fn)

print('Test')
test_loop(model, dataloaders['test'], loss_fn)
print("Done!")
