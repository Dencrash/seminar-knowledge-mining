from .feature import Feature


class SizeFeature(Feature):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self):
        yield self.width
        yield self.height
        yield self.width / self.height

    @classmethod
    def names(cls):
        yield 'external_width'
        yield 'external_height'
        yield 'external_ratio'
