from torch.utils.data import DataLoader

from Experiments.mynet.net import M2FaVMUnet
from datasets.dataset import TN3K
from tensorboardX import SummaryWriter
from Experiments.engine import *
import sys
from Experiments.utils import *
from Experiments.config.config_setting import setting_config

def main(config):
    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    flod = 1
    train_loader_folds = []
    val_loader_folds = []
    for i in range(flod):
        train_set = TN3K(config.data_path, config, mode='train')
        val_set = TN3K(config.data_path, config, mode='val')
        train_loader = DataLoader(train_set,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=config.num_workers)
        val_loader = DataLoader(val_set,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
        train_loader_folds.append(train_loader)
        val_loader_folds.append(val_loader)
    test_set = TN3K(config.data_path, config, mode='test')
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)

    for fold in range(1):  # 遍历每个折叠
        print(f"#--------------开始折 {fold + 1}/{flod}------------------#")
        # 获取当前折的数据加载器
        train_loader = train_loader_folds[fold]
        val_loader = val_loader_folds[fold]
        print('#----------Prepareing Model----------#')
        # 初始化模型、优化器、损失函数
        model = Baseline(in_channels=3, out_channels=1)
        model =  M2FaVMUnet(in_channels=3,out_channels=1)
        model = model.cuda()

        print('#----------Prepareing loss, opt, sch and amp----------#')
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        print('#----------Set other params----------#')
        min_loss = 999
        start_epoch = 1
        min_epoch = 1
        step = 0

        print('#----------Creating logger----------#')
        sys.path.append(config.work_dir + '/')
        log_dir = os.path.join(config.work_dir, f'log/fold{fold + 1}')
        checkpoint_dir = os.path.join(config.work_dir, f'checkpoints/fold{fold + 1}')
        outputs = os.path.join(config.work_dir, f'outputs/fold{fold + 1}')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        global logger
        logger = get_logger('train', log_dir)
        global writer
        writer = SummaryWriter(config.work_dir + f'summary/fold{fold + 1}')

        log_config_info(config, logger)

        print('#----------Training----------#')

        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()

            step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                config,
                writer,
                fold
            )

            loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                fold
            )

            if loss < min_loss:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(checkpoint_dir + '/best.pth', map_location=torch.device('cpu'))
            model.load_state_dict(best_weight)
            loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                fold
            )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'fold-{fold + 1}--best-epoch{min_epoch}-loss{min_loss:.4f}.pth'))


if __name__ == '__main__':
    config = setting_config
    main(config)
