def adjust_weight_decay(optimizer, epoch, args):
    weight_decay = epoch / args.epochs * args.weight_decay

    for param_group in optimizer.param_groups:
        # only update the param_group with weight_decay  > 0.0
        if param_group["weight_decay"] > 0.:
            param_group['weight_decay'] = weight_decay
    return weight_decay