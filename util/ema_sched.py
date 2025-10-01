def adjust_ema_decay(epoch, args):
    ema_decay = epoch / args.epochs * (1.0 - args.ema_rate) + args.ema_rate
    return min(ema_decay, 0.9999)
