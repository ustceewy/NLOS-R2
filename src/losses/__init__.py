from .cls import CELoss
from .common import L1Loss, MSELoss


def build_loss(opt_loss, logger):
    """Build loss from options.
    """

    loss_type = opt_loss.pop('type')
    loss = eval(loss_type)(**opt_loss)
    logger.write(f'Loss {loss_type} is created')
    
    return loss
