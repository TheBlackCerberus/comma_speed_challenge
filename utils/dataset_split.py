from torch.utils.data import Dataset


def create_train_val_test_split(
    dataset: Dataset, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2
) -> tuple[list[int], list[int], list[int]]:
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_end = int(dataset_size * train_ratio)
    val_end = train_end + int(dataset_size * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices
