from .xml_style import XMLDataset
class IDD(XMLDataset):
    CLASSES =( ' person', 'rider',' car' ,'truck','bus','motorcycle','bicycle','autorickshaw','animal','traffic light','traffic sign', 'vehicle fallback','caravan', 'trailer', 'train')

    def __init__(self, **kwargs):
        super(IDD, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')