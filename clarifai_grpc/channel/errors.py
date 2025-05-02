class ApiError(Exception):
    pass


class NotImplementedCaller:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class UsageError(Exception):
    pass
