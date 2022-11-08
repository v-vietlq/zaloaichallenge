def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()
