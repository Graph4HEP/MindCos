#necessary packages
import os, time, sys
#mindspore packages
import mindspore as ms

def process_bar_train(num, total, dt, loss, acc, Type=''):
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    r = '\r{} [{}{}]{}/{} - used {:.1f}s / left {:.1f}s / loss {:.10f} / acc {:.4f} '.format(Type, '*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, loss, acc)
    sys.stdout.write(r)
    sys.stdout.flush()

def forward_fn(model, loss_fn, nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, label):
    logits = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
    loss = loss_fn(logits, label)
    return loss, logits

def train_loop(model, dataloader, loss_fn, grad_fn, optimizer, ep, filename):
    num_batches = len(dataloader)
    model.set_train()
    st = time.time()
    total, loss, correct = 0, 0, 0    
    with ms.SummaryRecord(f'./summary_dir/{filename}', network=model) as summary_record:
        for i, (label, p4s, nodes, atom_mask, edge_mask, edges) in enumerate(dataloader):
            label = label.astype(ms.int32)
            p4s = p4s.astype(ms.float32)
            nodes = nodes.astype(ms.float32)
            atom_mask = atom_mask.astype(ms.float32)
            edge_mask = edge_mask.astype(ms.float32)
            edges = edges.astype(ms.int32)
            batch_size, n_nodes, _ = p4s.shape
            atom_positions = p4s.reshape(batch_size * n_nodes, -1)
            atom_mask = atom_mask.reshape(batch_size * n_nodes, -1)
            edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1)
            nodes = nodes.reshape(batch_size * n_nodes, -1)
            (step_loss, logits), grads = grad_fn(model, loss_fn, nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, label)
            optimizer(grads)             
            loss += step_loss
            correct += (logits.argmax(1) == label).asnumpy().sum()
            total += len(p4s)
            summary_record.add_value('scalar', 'loss', loss/(i+1))
            summary_record.add_value('scalar', 'accuracy', ms.Tensor(100*correct/total))
            summary_record.record(i+1+ep*num_batches)            
            process_bar_train(i+1, num_batches, time.time()-st, loss.asnumpy()/(i+1), 100*correct/total, '')
    os.makedirs(f'checkpoint/{filename}', exist_ok=True)
    ms.save_checkpoint(model, f'checkpoint/LorentzNet_ep{ep}.ckpt')

def test_loop(model, dataloader, loss_fn):
    num_batches =len(dataloader)
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for i, (label, p4s, nodes, atom_mask, edge_mask, edges) in enumerate(dataloader):
        label = label.astype(ms.int32)
        p4s = p4s.astype(ms.float32)
        nodes = nodes.astype(ms.float32)
        atom_mask = atom_mask.astype(ms.float32)
        edge_mask = edge_mask.astype(ms.float32)
        edges = edges.astype(ms.int32)
        batch_size, n_nodes, _ = p4s.shape
        atom_positions = p4s.reshape(batch_size * n_nodes, -1)
        atom_mask = atom_mask.reshape(batch_size * n_nodes, -1)
        edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1)
        nodes = nodes.reshape(batch_size * n_nodes, -1)
        pred = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
        total += len(p4s)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Valid: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
