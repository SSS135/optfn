def get_param_groups(model, exclude_single_dim=True):
    bias_group = []
    others_group = []
    for n, p in model.named_parameters():
        if n.find('bias') != -1 or n.find('beta') != -1 or (exclude_single_dim and p.dim() == 1):
            bias_group.append(p)
        else:
            others_group.append(p)
    return [dict(params=others_group),
            dict(params=bias_group, weight_decay=0)]