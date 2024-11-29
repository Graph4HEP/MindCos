#necessary packages
import os, time, sys
#mindspore packages
import mindspore 
from mindspore import ops, Tensor

def process_bar_train(num, total, dt, loss, acc, Type=''):
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    r = '\r{} [{}{}]{}/{} - used {:.1f}s / left {:.1f}s / loss {:.10f} / acc {:.4f} '.format(Type, '*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, loss, acc)
    sys.stdout.write(r)
    sys.stdout.flush()

def forward_fn(net, criterion, node_feat, edge_feat, src, dst, edge_label):
    logits = net(node_feat, edge_feat, src, dst)
    edge_label = mindspore.ops.squeeze(edge_label)
    loss = criterion(logits, edge_label)
    return loss, logits

def train_loop(net, dataloader, criterion, grad_fn, optimizer, ep, filename):
    total_n_data, total_loss, total_correct = 0, 0, 0
    total_n_recon_link, total_n_truth_link = 0, 0

    net.set_train()
    st = time.time()
    with mindspore.SummaryRecord(f'./summary_dir/{filename}_per_steps', network=net) as summary_record:
        for i in range(dataloader['len']):
            (loss, pred), grads = grad_fn(net, criterion,
                                          Tensor(dataloader['node_feat'][i], dtype=mindspore.float32),
                                          Tensor(dataloader['edge_feat'][i], dtype=mindspore.float32),
                                          Tensor(dataloader['bs_src'][i], dtype=mindspore.float32),
                                          Tensor(dataloader['bs_dst'][i], dtype=mindspore.float32),
                                          Tensor(dataloader['edge_label'][i], dtype=mindspore.float32)
                                         )            
            optimizer(grads)
            label = ops.squeeze(Tensor(dataloader['edge_label'][i], dtype=mindspore.float32))
            predict = ops.Select()(pred>0.5, Tensor(ops.ones_like(pred), pred.dtype), Tensor(ops.zeros_like(pred), pred.dtype))
            correct = ops.where(predict==label, Tensor(ops.ones_like(predict), predict.dtype), Tensor(ops.zeros_like(predict), predict.dtype)).sum().asnumpy()
            n_recon_link = ops.where(predict==1, Tensor(ops.ones_like(predict), predict.dtype), Tensor(ops.zeros_like(predict), predict.dtype)).sum().asnumpy()
            n_truth_link = ops.where(label==1, Tensor(ops.ones_like(predict), predict.dtype), Tensor(ops.zeros_like(predict), predict.dtype)).sum().asnumpy()
            total_n_recon_link += n_recon_link
            total_n_truth_link += n_truth_link
            total_n_data += len(predict)
            total_correct += correct
            total_loss += loss.asnumpy()
            process_bar_train(i+1, dataloader['len'], time.time()-st, total_loss/total_n_data, total_correct/total_n_data, '')
            summary_record.add_value('scalar', 'loss',  Tensor(total_loss/total_n_data))
            summary_record.add_value('scalar', 'accuracy', Tensor(total_correct/total_n_data))
            summary_record.record(i+1+ep*dataloader['len'])
    with mindspore.SummaryRecord(f'./summary_dir/{filename}_per_epoch', network=net) as summary_record:        
        summary_record.add_value('scalar', 'loss',  Tensor(total_loss/total_n_data))
        summary_record.add_value('scalar', 'accuracy', Tensor(total_correct/total_n_data))
        summary_record.record(ep)
    os.makedirs(f'checkpoint/{filename}', exist_ok=True)
    mindspore.save_checkpoint(net, f'checkpoint/TrackFinding_ep{ep}.ckpt')

def test_loop(net, dataloader, criterion):
    net.set_train(False)
    total_n_data, total_loss, total_correct = 0, 0, 0
    total_n_recon_link, total_n_truth_link = 0, 0
    for i in range(dataloader['len']):
        pred = net(Tensor(dataloader['node_feat'][i], dtype=mindspore.float32), 
                   Tensor(dataloader['edge_feat'][i], dtype=mindspore.float32),
                   Tensor(dataloader['bs_src'][i], dtype=mindspore.float32),
                   Tensor(dataloader['bs_dst'][i], dtype=mindspore.float32))
        edge_label = ops.squeeze(Tensor(dataloader['edge_label'][i], dtype=mindspore.float32)) 
        loss = criterion(pred, edge_label)
        total_n_data += len(pred)
        predict = ops.Select()(pred>0.5, Tensor(ops.ones_like(pred), pred.dtype), Tensor(ops.zeros_like(pred), pred.dtype))
        correct = ops.where(predict==edge_label, Tensor(ops.ones_like(predict), predict.dtype), Tensor(ops.zeros_like(predict), predict.dtype)).sum().asnumpy()
        total_correct += correct
        total_loss += loss.asnumpy()
    total_loss /= total_n_data
    total_correct /= total_n_data
    print(f"\nValid: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")


        
