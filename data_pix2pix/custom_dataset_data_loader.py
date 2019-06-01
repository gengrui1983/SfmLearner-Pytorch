import torch.utils.data

from data_pix2pix.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'sequence_folders':
        from datasets.sequence_folders import SequenceFolder
        dataset = SequenceFolder(opt.root, opt.segmentation, pix2pix=True, p2p_benchmark=opt.pix2pix_benchmark)
    else:
        if opt.dataset_mode == 'aligned':
            from data_pix2pix.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
        elif opt.dataset_mode == 'unaligned':
            from data_pix2pix.unaligned_dataset import UnalignedDataset
            dataset = UnalignedDataset()
        elif opt.dataset_mode == 'single':
            from data_pix2pix.single_dataset import SingleDataset
            dataset = SingleDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

        print("dataset [%s] was created" % (dataset.name()))
        dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
