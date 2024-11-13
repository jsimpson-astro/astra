class dummy_pbar():
    """
    Placeholder for tqdm pbar that does nothing
    """
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, num, *args, **kwargs):
        pass