import torch
def kmeans(x, ncluster, niter=5):
    '''
    From minGPT implement
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    B, N, D = x.size()
    c = x[:, torch.randperm(N, device=x.device)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        coord_diff = (x[:, :, None, :] - c[:, None, :, :])**2
        min_id = coord_diff.sum(-1).argmin(-1)
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        coord_ = coord_diff.new_zeros(coord_diff.size())
        # coord_ = torch.zeros(coord_diff.size(),device=)
        coord_.scatter_(-2, min_id[:, :, None, None].repeat(1,1,1,3), x[:,:,None,:])
        coord_sum = coord_.sum(1)
        corrd_nozero_count = (coord_!=0).any(-1).sum(1)
        c = coord_sum / corrd_nozero_count[:, :, None]
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=-1)
        # ndead = nanix.sum()
        # print('done step {}/{}, re-initialized {} dead clusters'.format(i+1, niter, ndead))
        c[nanix] = x[:, torch.randperm(N, device=x.device)][:, :ncluster, :][nanix]
    return c
