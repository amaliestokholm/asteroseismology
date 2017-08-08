import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
#import seaborn as sns


class MyScalarFormatter(ScalarFormatter):
    def get_offset(self):
        a = self.get_scale_string()
        b = self.get_offset_string()
        if a or b:
            return '$%s$' % self.fix_minus(a + b)
        return ''

    def get_scale_string(self):
        if self.orderOfMagnitude:
            return r'\times 10^{%.0f}' % self.orderOfMagnitude
        else:
            return ''

    def get_offset_string(self):
        if self.offset:
            s = '%+g' % self.offset
            if 'e' in s:
                b, e = s.split('e')
                return r'%s \times 10^{%s}' % (b, e.lstrip('+0'))
            else:
                return s
        else:
            return ''
