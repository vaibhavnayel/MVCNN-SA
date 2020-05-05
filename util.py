import torch
import os

def logEpoch(logger, model, epoch, valloss, valaccuracy):
    # 1. Log scalar values (scalar summary)
    info = {'val loss': valloss,
            # 'val loss': valloss.item(), 
        'val accuracy': valaccuracy,
        # 'val accuracy': valaccuracy.item(),

        # 'train loss':trainloss.item(),
        # 'train accuracy':trainaccuracy.item()
        }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    # for tag, value in model.named_parameters():
    #     tag = tag.replace('.', '/')
    #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
    #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
    #info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

    #for tag, images in info.items():
        #logger.image_summary(tag, images, epoch)

def save_checkpoint(state, model, start_time, depth, checkpoint_dir='checkpoint', filename='checkpoint.pth.tar'):
    if model not in ['resnet','resnet_att']:depth=None
    filepath = os.path.join(checkpoint_dir, model +depth+ '_' +start_time + '_' + filename)
    torch.save(state, filepath)
