def align_loss(class_logits, labels, device=torch.device('cuda:0')): # score: [64, 14] labels: tensor([9, 4, 12 ,... 7, 7, 7]
    # 0.07 为softmax的温度系数 Ref: https://blog.csdn.net/weixin_43845922/article/details/133964299
    n_feat = class_logits[:,7].unsqueeze(1) # [64, 1]
    a_feat = torch.cat((class_logits[:, :7], class_logits[:, 8:]), dim=1)
    loss_hinge = torch.mean(torch.mean(F.relu(1 - (n_feat - a_feat)), dim=1)) # 计算每行的 Hinge Loss
    return CE_loss(class_logits / 0.07, labels) + loss_hinge



def generate_TAN_label(mask, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    mask: 输入的mask, 例如 mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1,],
                                          [0, 1, 1, 0, 1, 1, 1,]])
    return: 返回label矩阵，shape：[batch_size, len(mask[0], len(mask[0])]
    """
    
    def generate_box(mask, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        rectangles = []
        in_rectangle = False
        for i, value in enumerate(mask):
            # 如果当前位置是矩形的开始位置，且不在矩形内部
            if value == 1 and not in_rectangle:
                start_position = i  # 记录矩形的开始位置
                in_rectangle = True  # 表示进入矩形内部
            # 如果当前位置是矩形的结束位置，且在矩形内部
            elif value == 0 and in_rectangle:
                end_position = i - 1  # 记录矩形的结束位置
                rectangles.append((start_position, end_position))  # 将矩形的坐标加入列表
                in_rectangle = False  # 表示离开矩形内部

        # 处理矩形延伸到mask末尾的情况
        if in_rectangle:
            rectangles.append((start_position, len(mask) - 1))
        
        # 将生成的bounding box列表填充到矩阵中
        matrix = torch.zeros((len(mask), len(mask)), dtype=torch.float).to(device)
        for rectangle in rectangles:
            matrix[rectangle[0], rectangle[1]] = 1
        # print(rectangles)
        return matrix
    
    label_matrix = []
    for batch in mask:
       matrix = generate_box(batch, device) # [32, 32]
       label_matrix.append(matrix.unsqueeze(0))
    label_matrix = torch.cat(label_matrix, dim=0)
    return label_matrix

def mid_loss(score, labels, class_score_map2d, class_score_map2d_normal, class_score_map2d_abnormal, class_score_map2d_abnormal_merge):
    """
    score: [64, 32, 1]
    labels: labels: tensor([9, 4, 12 ,... 7, 7, 7]
    class_score_map2d: [64, 14, 32, 32]
    class_score_map2d_normal: [64, 1, 32, 32]
    class_score_map2d_abnormal: [64, 13, 32, 32]
    class_score_map2d_abnormal_merge: [64, 1, 32, 32] 
    """
    # score = score.squeeze() # [64, 32]
    # mask_ano = (score > 0.5).int()
    # mask_nor = (score < 0.5).int()
    # # 
    # score_TAN_ano = generate_TAN_label(mask_ano) # [64, 32, 32]
    # score_TAN_nor = generate_TAN_label(mask_nor) # [64, 32, 32]

    # bce_loss_ano = binary_CE_loss(input=class_score_map2d_abnormal_merge.squeeze(), target=score_TAN_ano)
    # bce_loss_nor = binary_CE_loss(input=class_score_map2d_normal.squeeze(), target=score_TAN_nor)
    # loss = bce_loss_ano + bce_loss_nor
    # print(score_TAN_ano.shape)

    multi_class, _ = torch.topk(class_score_map2d.reshape(class_score_map2d.shape[0], class_score_map2d.shape[1], -1), k=1, dim=-1)
    loss_ce = CE_loss(multi_class.squeeze(-1) / 0.07 , labels)

    loss = loss_ce * 1.0
    return loss
