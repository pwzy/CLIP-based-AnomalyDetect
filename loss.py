def align_loss(class_logits, labels, device=torch.device('cuda:0')): # score: [64, 14] labels: tensor([9, 4, 12 ,... 7, 7, 7]
    # 0.07 为softmax的温度系数 Ref: https://blog.csdn.net/weixin_43845922/article/details/133964299
    n_feat = class_logits[:,7].unsqueeze(1) # [64, 1]
    a_feat = torch.cat((class_logits[:, :7], class_logits[:, 8:]), dim=1)
    loss_hinge = torch.mean(torch.mean(F.relu(1 - (n_feat - a_feat)), dim=1)) # 计算每行的 Hinge Loss
    return CE_loss(class_logits / 0.07, labels) + loss_hinge
